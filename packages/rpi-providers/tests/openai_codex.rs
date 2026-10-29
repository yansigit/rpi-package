//! OpenAI Codex (ChatGPT subscription) through the real cdylib: catalog, OAuth begin and the
//! state check, the token endpoint, the Codex Responses request, cancellation, rate limits and
//! credential redaction. Every request goes to a loopback server; nothing reaches OpenAI.
use std::{path::PathBuf, sync::Arc, time::Duration};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rpi_ai::{
    AssistantMessageEvent, Content, Context, Message, Model, Provider, SimpleStreamOptions,
    StopReason, ThinkingLevel, UserMessage,
};
use rpi_providers::openai_codex::{auth, PROVIDER_ID};
use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};
use url::Url;

const PI_ORIGIN: &str = "https://chatgpt.com";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

fn library_path() -> PathBuf {
    let profile = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let filename = format!(
        "{}rpi_providers{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    [profile.join(&filename), profile.join("deps").join(filename)]
        .into_iter()
        .find(|path| path.is_file())
        .expect("first cargo build -p rpi-providers")
}

fn load() -> (rpi_extensions::ExtensionSession, Arc<dyn Provider>) {
    let session = rpi_extensions::load_session_mixed(
        &[],
        &[library_path()],
        Arc::new(rpi_extensions::NullDiagnostics),
        None,
    );
    let provider = rpi_extensions::PluggableProvider::from_session(
        &session,
        tokio::runtime::Handle::current(),
    )
    .into_iter()
    .find(|provider| provider.id() == PROVIDER_ID)
    .expect("openai-codex is registered");
    (session, provider)
}

/// An unsigned JWT whose payload carries the ChatGPT account claim, as OpenAI issues it.
fn jwt(account: &str) -> String {
    let payload = json!({"https://api.openai.com/auth": {"chatgpt_account_id": account}});
    format!(
        "eyJhbGciOiJub25lIn0.{}.fixture-signature",
        URL_SAFE_NO_PAD.encode(payload.to_string())
    )
}

fn model(provider: &dyn Provider, id: &str, origin: &str) -> Model {
    let mut model = provider
        .models()
        .iter()
        .find(|model| model.id == id)
        .unwrap_or_else(|| panic!("{id} in catalog"))
        .clone();
    assert_eq!(model.base_url, "https://chatgpt.com/backend-api");
    model.base_url = model.base_url.replacen(PI_ORIGIN, origin, 1);
    model
}

fn options(key: Option<&str>) -> SimpleStreamOptions {
    SimpleStreamOptions {
        api_key: key.map(str::to_string),
        session_id: Some("session-1".into()),
        max_retries: Some(0),
        timeout: Some(Duration::from_secs(5)),
        ..Default::default()
    }
}

fn context() -> Context {
    Context::new(vec![Message::User(UserMessage::new("hello", 0))])
}

/// Reads one request; the body is returned as text so form and JSON bodies both fit.
async fn read_request(socket: &mut TcpStream) -> (String, String) {
    let mut bytes = Vec::new();
    loop {
        let mut buffer = [0; 4096];
        let count = socket.read(&mut buffer).await.unwrap();
        assert_ne!(count, 0);
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
            let length: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            if bytes.len() >= end + 4 + length {
                let body = String::from_utf8_lossy(&bytes[end + 4..end + 4 + length]).into();
                return (head, body);
            }
        }
    }
}

async fn serve(
    status: u16,
    kind: &'static str,
    body: String,
) -> (String, JoinHandle<(String, String)>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        request
    });
    (origin, server)
}

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect()
}

fn responses_reply(text: &str) -> String {
    let item = json!({"type":"message","id":"msg_1","role":"assistant","status":"completed",
        "content":[{"type":"output_text","text":text,"annotations":[]}]});
    sse(&[
        json!({"type":"response.output_item.added","output_index":0,
            "item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}),
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,
            "content_index":0,"delta":text}),
        json!({"type":"response.output_item.done","output_index":0,"item":item}),
        json!({"type":"response.completed","response":{"id":"resp_1","status":"completed",
            "output":[item],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}),
    ])
}

fn header_values<'a>(head: &'a str, name: &str) -> Vec<&'a str> {
    head.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
        .collect()
}

async fn stream_once(
    provider: &dyn Provider,
    id: &str,
    options: &SimpleStreamOptions,
    status: u16,
    kind: &'static str,
    reply: String,
) -> (rpi_ai::AssistantMessage, String, String) {
    let (origin, server) = serve(status, kind, reply).await;
    let stream = provider
        .stream_simple(&model(provider, id, &origin), &context(), options)
        .await;
    let message = tokio::time::timeout(Duration::from_secs(5), stream.result())
        .await
        .unwrap()
        .unwrap();
    let (head, body) = server.await.unwrap();
    (message, head, body)
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[tokio::test]
async fn catalog_is_pis_codex_models_with_oauth_only() {
    let (session, provider) = load();
    let snapshot = session.snapshot().unwrap();
    let manifest = snapshot.provider_manifest(PROVIDER_ID).cloned().unwrap();
    assert_eq!(
        manifest.api_key_env, None,
        "OAuth only: no API-key variable"
    );
    assert!(manifest.requires_auth);
    let ids: Vec<_> = provider.models().iter().map(|m| m.id.as_str()).collect();
    // Pi 1.3.0 order, Pi's default (gpt-6.1-sol) first.
    assert_eq!(
        ids,
        [
            "gpt-6.1-sol",
            "gpt-5.3-codex-spark",
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-astra",
            "gpt-6-luna",
            "gpt-6-sol",
        ]
    );
    assert!(manifest.models.iter().all(|m| m.provider == PROVIDER_ID
        && m.api == rpi_ai::Api::OpenaiResponses
        && m.cost_configured));
    let oauth = rpi_extensions::list_oauth_providers(snapshot);
    assert!(oauth.iter().any(|provider| provider.id == PROVIDER_ID));
}

#[tokio::test]
async fn oauth_begin_issues_pis_pkce_url_and_exchange_checks_the_state() {
    let (session, _provider) = load();
    let snapshot = session.snapshot_arc().unwrap();
    let response = rpi_extensions::request_oauth(&snapshot, PROVIDER_ID, "begin", json!({}))
        .expect("begin needs no configuration: the client is public");
    assert_eq!(response["kind"], "interaction");
    let login = Url::parse(response["payload"]["display"]["url"].as_str().unwrap()).unwrap();
    assert_eq!(
        login.origin().ascii_serialization(),
        "https://auth.openai.com"
    );
    assert_eq!(login.path(), "/oauth/authorize");
    let query: std::collections::HashMap<_, _> = login.query_pairs().into_owned().collect();
    assert_eq!(query["response_type"], "code");
    assert_eq!(query["client_id"], CLIENT_ID);
    assert_eq!(query["redirect_uri"], "http://localhost:1455/auth/callback");
    assert_eq!(query["scope"], "openid profile email offline_access");
    assert_eq!(query["code_challenge_method"], "S256");
    assert_eq!(query["code_challenge"].len(), 43, "S256 digest, base64url");
    assert_eq!(query["id_token_add_organizations"], "true");
    assert_eq!(query["codex_cli_simplified_flow"], "true");
    assert_eq!(query["originator"], "pi");
    // Pi's state: 16 random bytes in hex.
    let state = &query["state"];
    assert_eq!(state.len(), 32);
    assert!(state.chars().all(|c| c.is_ascii_hexdigit()));

    // A redirect from another login is refused before any request is sent.
    let foreign = "http://localhost:1455/auth/callback?code=fixture-code&state=other";
    let error = rpi_extensions::request_oauth(
        &snapshot,
        PROVIDER_ID,
        "exchange",
        json!({"value": foreign}),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("State mismatch"), "{error}");
    assert!(!error.contains("fixture-code"), "{error}");
    let error = rpi_extensions::request_oauth(
        &snapshot,
        PROVIDER_ID,
        "exchange",
        json!({"value": format!("http://localhost:1455/auth/callback?state={state}")}),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("Missing authorization code"), "{error}");
}

#[test]
fn pasted_input_accepts_pis_four_shapes() {
    for (input, code, state) in [
        (
            "http://localhost:1455/auth/callback?code=c1&state=s1",
            Some("c1"),
            Some("s1"),
        ),
        ("c2#s2", Some("c2"), Some("s2")),
        ("code=c3&state=s3", Some("c3"), Some("s3")),
        ("  c4  ", Some("c4"), None),
        ("", None, None),
    ] {
        let parsed = auth::parse_input(input);
        assert_eq!(
            (parsed.0.as_deref(), parsed.1.as_deref()),
            (code, state),
            "{input}"
        );
    }
}

/// One token-endpoint reply; returns the form body the client posted.
async fn token_endpoint(status: u16, body: Value) -> (String, JoinHandle<(String, String)>) {
    let (origin, server) = serve(status, "application/json", body.to_string()).await;
    (format!("{origin}/oauth/token"), server)
}

#[tokio::test]
async fn exchange_and_refresh_store_the_account_with_the_access_token() {
    let client = reqwest::Client::new();
    let access = jwt("acct-1");
    let tokens =
        json!({"access_token": access, "refresh_token": "fixture-refresh", "expires_in": 3600});

    let (url, server) = token_endpoint(200, tokens.clone()).await;
    let envelope = auth::exchange(&client, &url, "fixture-code", "fixture-verifier")
        .await
        .unwrap();
    let (head, body) = server.await.unwrap();
    assert!(head.starts_with("post /oauth/token "), "{head}");
    assert_eq!(
        header_values(&head, "content-type"),
        ["application/x-www-form-urlencoded"]
    );
    let form: std::collections::HashMap<_, _> = url::form_urlencoded::parse(body.as_bytes())
        .into_owned()
        .collect();
    assert_eq!(form["grant_type"], "authorization_code");
    assert_eq!(form["client_id"], CLIENT_ID);
    assert_eq!(form["code"], "fixture-code");
    assert_eq!(form["code_verifier"], "fixture-verifier");
    assert_eq!(form["redirect_uri"], "http://localhost:1455/auth/callback");
    assert_eq!(envelope["kind"], "credential");
    let payload = &envelope["payload"];
    assert_eq!(payload["access"], format!("{access}|acct-1"));
    assert_eq!(payload["refresh"], "fixture-refresh");
    // Seconds on the host envelope, not Pi's milliseconds.
    let expires = payload["expires_at"].as_i64().unwrap();
    assert!(
        (now() + 3590..=now() + 3610).contains(&expires),
        "{expires}"
    );

    let (url, server) = token_endpoint(200, tokens).await;
    let envelope = auth::refresh(&client, &url, "fixture-refresh")
        .await
        .unwrap();
    let (_, body) = server.await.unwrap();
    let form: std::collections::HashMap<_, _> = url::form_urlencoded::parse(body.as_bytes())
        .into_owned()
        .collect();
    assert_eq!(form["grant_type"], "refresh_token");
    assert_eq!(form["refresh_token"], "fixture-refresh");
    assert_eq!(form["client_id"], CLIENT_ID);
    assert_eq!(envelope["payload"]["access"], format!("{access}|acct-1"));
}

#[tokio::test]
async fn token_failures_never_echo_credentials() {
    let client = reqwest::Client::new();
    // Pi requires the account claim.
    let (url, server) = token_endpoint(
        200,
        json!({"access_token": "not-a-jwt", "refresh_token": "fixture-refresh", "expires_in": 60}),
    )
    .await;
    let error = auth::exchange(&client, &url, "fixture-code", "v")
        .await
        .unwrap_err();
    server.await.unwrap();
    assert!(
        error.contains("Failed to extract accountId from token"),
        "{error}"
    );
    assert!(
        !error.contains("not-a-jwt") && !error.contains("fixture-refresh"),
        "{error}"
    );

    let (url, server) = token_endpoint(
        400,
        json!({"error": "invalid_grant", "error_description": "refresh token fixture-refresh was reused"}),
    )
    .await;
    let error = auth::refresh(&client, &url, "fixture-refresh")
        .await
        .unwrap_err();
    server.await.unwrap();
    assert!(
        error.contains("OpenAI Codex token refresh failed (400)"),
        "{error}"
    );
    assert!(error.contains("invalid_grant"), "{error}");
    assert!(!error.contains("fixture-refresh"), "{error}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_sends_pis_codex_request() {
    let (_session, provider) = load();
    let access = jwt("acct-1");
    let packed = format!("{access}|acct-1");
    let (message, head, body) = stream_once(
        provider.as_ref(),
        "gpt-6.1-sol",
        &options(Some(&packed)),
        200,
        "text/event-stream",
        responses_reply("hi"),
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{message:?}");
    assert_eq!(message.provider, PROVIDER_ID);
    assert_eq!(Content::text_only(&message.content, ""), "hi");
    assert!(
        head.starts_with("post /backend-api/codex/responses "),
        "{head}"
    );
    // Only the bare token travels; the packed account goes in its own header.
    assert_eq!(
        header_values(&head, "authorization"),
        [format!("bearer {}", access.to_ascii_lowercase())]
    );
    assert_eq!(header_values(&head, "chatgpt-account-id"), ["acct-1"]);
    assert_eq!(header_values(&head, "originator"), ["pi"]);
    assert_eq!(
        header_values(&head, "openai-beta"),
        ["responses=experimental"]
    );
    assert_eq!(header_values(&head, "accept"), ["text/event-stream"]);
    assert_eq!(header_values(&head, "session-id"), ["session-1"]);
    assert_eq!(header_values(&head, "x-client-request-id"), ["session-1"]);
    assert!(
        header_values(&head, "user-agent")[0].starts_with("pi ("),
        "{head}"
    );
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["model"], "gpt-6.1-sol");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["instructions"], "You are a helpful assistant.");
    assert_eq!(body["text"], json!({"verbosity": "low"}));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["prompt_cache_key"], "session-1");
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["parallel_tool_calls"], true);
    assert!(body.get("max_output_tokens").is_none(), "{body}");
    // gpt-6.1-sol maps "off" to null: no reasoning field when thinking is off.
    assert!(body.get("reasoning").is_none(), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_reads_the_account_from_a_bare_token_and_maps_reasoning_like_pi() {
    let (_session, provider) = load();
    let access = jwt("acct-from-jwt");
    for (level, effort) in [
        (None, json!({"effort": "none"})),
        (
            Some(ThinkingLevel::High),
            json!({"effort": "high", "summary": "auto"}),
        ),
        (
            Some(ThinkingLevel::Minimal),
            json!({"effort": "low", "summary": "auto"}),
        ),
    ] {
        let mut options = options(Some(&access));
        options.reasoning = level;
        let (message, head, body) = stream_once(
            provider.as_ref(),
            "gpt-5.5",
            &options,
            200,
            "text/event-stream",
            responses_reply("hi"),
        )
        .await;
        assert_eq!(message.stop_reason, StopReason::Stop, "{message:?}");
        assert_eq!(
            header_values(&head, "chatgpt-account-id"),
            ["acct-from-jwt"]
        );
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["reasoning"], effort, "{level:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_missing_or_unreadable_token_fails_before_any_request() {
    let (_session, provider) = load();
    for key in [None, Some("not-a-jwt")] {
        // Nothing listens here; an attempted request would report a connection error instead.
        let model = model(provider.as_ref(), "gpt-5.5", "http://127.0.0.1:9");
        let message = provider
            .stream_simple(&model, &context(), &options(key))
            .await
            .result()
            .await
            .unwrap();
        assert_eq!(message.stop_reason, StopReason::Error, "{key:?}");
        let error = message.error_message.unwrap_or_default();
        assert!(
            error.contains("rpi auth login --provider openai-codex")
                || error.contains("Failed to extract accountId from token"),
            "{error}"
        );
        assert!(!error.contains("not-a-jwt"), "{error}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_mid_stream_aborts_and_keeps_the_partial_text() {
    let (_session, provider) = load();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").await.unwrap();
        let item = json!({"type":"message","id":"msg_1","role":"assistant","status":"completed",
            "content":[{"type":"output_text","text":"partial","annotations":[]}]});
        let chunk = sse(&[
            json!({"type":"response.output_item.added","output_index":0,
                "item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}),
            json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,
                "content_index":0,"delta":"partial"}),
            json!({"type":"response.output_item.done","output_index":0,"item":item}),
        ]);
        socket
            .write_all(format!("{:x}\r\n{chunk}\r\n", chunk.len()).as_bytes())
            .await
            .unwrap();
        // Hold the response open until the client drops the connection.
        let mut buffer = [0; 64];
        let _ = socket.read(&mut buffer).await;
    });
    let options = options(Some(&jwt("acct-1")));
    let signal = options.signal.clone();
    let mut stream = provider
        .stream_simple(
            &model(provider.as_ref(), "gpt-5.5", &origin),
            &context(),
            &options,
        )
        .await;
    let mut last = None;
    while let Some(event) = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .unwrap()
    {
        if matches!(&event, AssistantMessageEvent::TextEnd { .. }) {
            signal.cancel();
        }
        last = Some(event);
    }
    let Some(AssistantMessageEvent::Error { reason, error }) = last else {
        panic!("expected an aborted terminal: {last:?}");
    };
    assert_eq!(reason, rpi_ai::ErrorReason::Aborted);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert!(!error
        .error_message
        .unwrap_or_default()
        .starts_with("rate_limited"));
    assert_eq!(Content::text_only(&error.content, ""), "partial");
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn usage_limits_become_the_rate_limit_marker_without_the_token() {
    let (_session, provider) = load();
    let access = jwt("acct-1");
    let packed = format!("{access}|acct-1");
    let soon = now() + 120;
    for (body, prefix) in [
        (
            json!({"error":{"type":"usage_limit_reached","message":format!("The usage limit has been reached for {access}"),
                "plan_type":"plus","resets_at":1893456000}}),
            "rate_limited; resets_at=1893456000 ".to_string(),
        ),
        (
            json!({"error":{"type":"usage_limit_reached","message":"limit","resets_in_seconds":120}}),
            "rate_limited; resets_at=".to_string(),
        ),
        (
            json!({"error":{"message":"Too Many Requests"}}),
            "rate_limited; ".to_string(),
        ),
    ] {
        let (message, ..) = stream_once(
            provider.as_ref(),
            "gpt-5.5",
            &options(Some(&packed)),
            429,
            "application/json",
            body.to_string(),
        )
        .await;
        assert_eq!(message.stop_reason, StopReason::Error);
        let error = message.error_message.unwrap_or_default();
        assert!(error.starts_with(&prefix), "{error}");
        assert!(!error.contains(&access), "token leaked: {error}");
        if body["error"]["resets_in_seconds"].is_number() {
            let at: i64 = error["rate_limited; resets_at=".len()..]
                .split(' ')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert!((soon - 5..=soon + 5).contains(&at), "{error}");
        }
        if body["error"]["resets_at"].is_null() && body["error"]["resets_in_seconds"].is_null() {
            assert!(!error.starts_with("rate_limited; resets_at"), "{error}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unauthorized_is_one_error_terminal_that_never_echoes_the_token() {
    let (_session, provider) = load();
    let access = jwt("acct-1");
    let packed = format!("{access}|acct-1");
    let body = json!({"error":{"message":format!("Invalid token: {access}")}}).to_string();
    let (origin, server) = serve(401, "application/json", body).await;
    let mut stream = provider
        .stream_simple(
            &model(provider.as_ref(), "gpt-5.5", &origin),
            &context(),
            &options(Some(&packed)),
        )
        .await;
    let mut terminals = 0;
    while let Some(event) = stream.next().await {
        terminals += usize::from(event.is_terminal());
    }
    assert_eq!(terminals, 1);
    let message = stream.result().await.unwrap();
    let error = message.error_message.unwrap_or_default();
    assert!(error.contains("401"), "{error}");
    assert!(!error.starts_with("rate_limited"), "{error}");
    assert!(!error.contains(&access), "{error}");
    server.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires the matching host in RPI_TEST_CLI; no network"]
async fn host_lists_checks_auth_and_streams_through_codex() {
    let host = std::env::var_os("RPI_TEST_CLI").expect("set RPI_TEST_CLI");
    let root = tempfile::tempdir().unwrap();
    let agent = root.path().join("agent");
    std::fs::create_dir(&agent).unwrap();
    let run = |args: &[&str]| {
        std::process::Command::new(&host)
            .current_dir(root.path())
            .env("RPI_CODING_AGENT_DIR", &agent)
            .env("RPI_OFFLINE", "1")
            .env_remove("OPENAI_API_KEY")
            .args(args)
            .arg("--extension")
            .arg(library_path())
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap()
    };
    let listed = run(&["--list-models", "openai-codex"]);
    let stdout = String::from_utf8_lossy(&listed.stdout);
    assert!(
        listed.status.success(),
        "{}",
        String::from_utf8_lossy(&listed.stderr)
    );
    assert!(
        stdout.contains("gpt-6.1-sol") && stdout.contains("gpt-5.5"),
        "{stdout}"
    );
    let checked = run(&["auth", "check", "--provider", PROVIDER_ID, "--json"]);
    let value: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(value["provider"], PROVIDER_ID);
    assert_eq!(value["ready"], false);
    assert_eq!(value["source"], "none");

    // A stored token streams through the host to Pi's path with the account header.
    let access = jwt("acct-host");
    let (origin, server) = serve(200, "text/event-stream", responses_reply("via codex")).await;
    std::fs::write(
        agent.join("models.json"),
        json!({"providers":{PROVIDER_ID:{"baseUrl":format!("{origin}/backend-api"),
            "apiKey":format!("{access}|acct-host"),"models":[]}}})
        .to_string(),
    )
    .unwrap();
    let printed = run(&[
        "--offline",
        "--package-only",
        "--no-session",
        "--no-builtin-tools",
        "--no-skills",
        "--no-context-files",
        "--no-prompt-templates",
        "--provider",
        PROVIDER_ID,
        "--model",
        "gpt-5.5",
        "--print",
        "hello",
    ]);
    let stdout = String::from_utf8_lossy(&printed.stdout);
    let stderr = String::from_utf8_lossy(&printed.stderr);
    assert!(printed.status.success(), "{stderr}");
    assert!(stdout.contains("via codex"), "{stdout}");
    assert!(!stdout.contains(&access) && !stderr.contains(&access));
    let (head, _) = server.await.unwrap();
    assert!(
        head.starts_with("post /backend-api/codex/responses "),
        "{head}"
    );
    assert_eq!(header_values(&head, "chatgpt-account-id"), ["acct-host"]);
}
