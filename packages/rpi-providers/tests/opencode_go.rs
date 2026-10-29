//! OpenCode Go through the real cdylib: catalog, per-model API routing, the session header,
//! auth, cancellation and HTTP errors. Every request goes to a loopback server.
use std::{path::PathBuf, sync::Arc, time::Duration};

use rpi_ai::{
    Api, AssistantMessageEvent, Content, Context, Message, Model, Provider, SimpleStreamOptions,
    StopReason, UserMessage,
};
use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};

const ID: &str = "opencode-go";
const KEY: &str = "fixture-opencode-key";
const PI_ORIGIN: &str = "https://opencode.ai";

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
    .find(|provider| provider.id() == ID)
    .expect("opencode-go is registered");
    (session, provider)
}

/// The catalog model with Pi's origin swapped for the loopback one, so the path is Pi's.
fn model(provider: &dyn Provider, id: &str, origin: &str) -> Model {
    let mut model = provider
        .models()
        .iter()
        .find(|model| model.id == id)
        .unwrap_or_else(|| panic!("{id} in catalog"))
        .clone();
    assert!(model.base_url.starts_with(PI_ORIGIN), "{}", model.base_url);
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

async fn read_request(socket: &mut TcpStream) -> (String, Value) {
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
                let body = serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
                return (head, body);
            }
        }
    }
}

/// One request, one complete response; returns the request head and body.
async fn serve(status: u16, body: String) -> (String, JoinHandle<(String, Value)>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        let kind = if status == 200 {
            "text/event-stream"
        } else {
            "application/json"
        };
        socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        request
    });
    (origin, server)
}

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| match event["type"].as_str() {
            // Anthropic and Responses frames carry an event name; chat chunks have no type.
            Some(kind) => format!("event: {kind}\ndata: {event}\n\n"),
            None => format!("data: {event}\n\n"),
        })
        .collect()
}

fn completions_reply(text: &str) -> String {
    sse(&[json!({"choices":[{"index":0,"delta":{"content":text},"finish_reason":"stop"}]})])
        + "data: [DONE]\n\n"
}

fn anthropic_reply(text: &str) -> String {
    sse(&[
        json!({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",
            "model":"fixture","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}),
        json!({"type":"message_stop"}),
    ])
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

/// One model per API family Pi routes, with the URL Pi's SDK client builds for it.
fn families() -> [(&'static str, Api, &'static str, String, String); 3] {
    [
        (
            "kimi-k3",
            Api::OpenaiCompletions,
            "/inference/go/v1/chat/completions",
            format!("authorization: bearer {KEY}"),
            completions_reply("hi"),
        ),
        (
            "claude-haiku-5-5",
            Api::AnthropicMessages,
            "/inference/go/v1/messages",
            format!("x-api-key: {KEY}"),
            anthropic_reply("hi"),
        ),
        (
            "gpt-6-luna",
            Api::OpenaiResponses,
            "/inference/go/v1/responses",
            format!("authorization: bearer {KEY}"),
            responses_reply("hi"),
        ),
    ]
}

fn header_values<'a>(head: &'a str, name: &str) -> Vec<&'a str> {
    head.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
        .collect()
}

#[tokio::test]
async fn catalog_is_pis_chat_models_that_rpi_ai_can_stream() {
    let (session, provider) = load();
    let manifest = session
        .snapshot()
        .unwrap()
        .provider_manifest(ID)
        .cloned()
        .unwrap();
    assert_eq!(manifest.api_key_env.as_deref(), Some("OPENCODE_API_KEY"));
    assert!(manifest.requires_auth);
    let ids: Vec<_> = provider.models().iter().map(|m| m.id.as_str()).collect();
    // Pi 1.3.0 order, Pi's default (kimi-k3) first; gemini-4-argon-contributor is dropped
    // because rpi-ai 0.8.3 has no google-generative-ai provider.
    assert_eq!(
        ids,
        [
            "kimi-k3",
            "claude-haiku-5-5",
            "minimax-m3",
            "qwen3.7-plus",
            "qwen3.8-flash",
            "qwen3.8-max",
            "cat-jam-free",
            "deepseek-v4-pro",
            "deepseek-v4.1-flash",
            "glm-5.2",
            "glm-5.3",
            "glm-5.3-flash",
            "hy3",
            "hy4-preview",
            "kimi-k2.7-code",
            "longcat-2.0",
            "longcat-2.5-preview-free",
            "mimo-v2.5",
            "mimo-v2.5-pro",
            "mimo-v2.6-flash",
            "mimo-v2.6-pro",
            "space-bunny",
            "step-5-preview-free",
            "gpt-5.6-luna",
            "gpt-6-luna",
            "grok-4.7",
            "muse-spark-1.2-contributor",
            "muse-spark-1.3-contributor",
        ]
    );
    // The host routes by (provider, api); every model must reach this plugin's one style.
    assert!(manifest
        .models
        .iter()
        .all(|m| m.provider == ID && m.api == Api::OpenaiCompletions && m.cost_configured));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_api_family_hits_pis_path_with_its_auth_and_session_headers() {
    let (_session, provider) = load();
    for (id, api, path, auth, reply) in families() {
        let (origin, server) = serve(200, reply).await;
        let stream = provider
            .stream_simple(
                &model(provider.as_ref(), id, &origin),
                &context(),
                &options(Some(KEY)),
            )
            .await;
        let message = tokio::time::timeout(Duration::from_secs(5), stream.result())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(message.stop_reason, StopReason::Stop, "{id}: {message:?}");
        assert_eq!(message.api, api, "{id}");
        assert_eq!(Content::text_only(&message.content, ""), "hi", "{id}");
        let (head, body) = server.await.unwrap();
        assert!(head.starts_with(&format!("post {path} ")), "{id}: {head}");
        assert!(head.lines().any(|line| line == auth), "{id}: {head}");
        assert_eq!(
            header_values(&head, "x-opencode-session"),
            ["session-1"],
            "{id}"
        );
        assert_eq!(body["model"], id);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_caller_session_header_wins_in_any_case() {
    let (_session, provider) = load();
    let (origin, server) = serve(200, completions_reply("hi")).await;
    let mut options = options(Some(KEY));
    options.headers = Some([("X-OpenCode-Session".into(), "caller".into())].into());
    let stream = provider
        .stream_simple(
            &model(provider.as_ref(), "kimi-k3", &origin),
            &context(),
            &options,
        )
        .await;
    assert_eq!(stream.result().await.unwrap().stop_reason, StopReason::Stop);
    let (head, _) = server.await.unwrap();
    assert_eq!(header_values(&head, "x-opencode-session"), ["caller"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_missing_key_is_a_clear_error_before_any_request() {
    let (_session, provider) = load();
    for (id, ..) in families() {
        // Nothing listens here; an attempted request would report a connection error instead.
        let model = model(provider.as_ref(), id, "http://127.0.0.1:9");
        let message = provider
            .stream_simple(&model, &context(), &options(None))
            .await
            .result()
            .await
            .unwrap();
        assert_eq!(message.stop_reason, StopReason::Error, "{id}");
        let error = message.error_message.unwrap_or_default();
        assert!(error.contains("OPENCODE_API_KEY"), "{id}: {error}");
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
        let chunk = sse(&[
            json!({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",
                "model":"fixture","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}),
            json!({"type":"content_block_stop","index":0}),
        ]);
        socket
            .write_all(format!("{:x}\r\n{chunk}\r\n", chunk.len()).as_bytes())
            .await
            .unwrap();
        // Hold the response open until the client drops the connection.
        let mut buffer = [0; 64];
        let _ = socket.read(&mut buffer).await;
    });
    let options = options(Some(KEY));
    let signal = options.signal.clone();
    let mut stream = provider
        .stream_simple(
            &model(provider.as_ref(), "claude-haiku-5-5", &origin),
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
    assert_eq!(Content::text_only(&error.content, ""), "partial");
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unauthorized_is_one_error_terminal_that_never_echoes_the_key() {
    let (_session, provider) = load();
    for (id, ..) in families() {
        // Some gateways echo the rejected key; it must not reach the error text.
        let body = json!({"error":{"message":format!("Invalid API key: {KEY}")}}).to_string();
        let (origin, server) = serve(401, body).await;
        let mut stream = provider
            .stream_simple(
                &model(provider.as_ref(), id, &origin),
                &context(),
                &options(Some(KEY)),
            )
            .await;
        let mut terminals = 0;
        while let Some(event) = stream.next().await {
            terminals += usize::from(event.is_terminal());
        }
        assert_eq!(terminals, 1, "{id}");
        let message = stream.result().await.unwrap();
        assert_eq!(message.stop_reason, StopReason::Error, "{id}");
        let error = message.error_message.unwrap_or_default();
        assert!(error.contains("401"), "{id}: {error}");
        assert!(!error.contains(KEY), "{id}: {error}");
        server.await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires the matching host in RPI_TEST_CLI; loopback HTTP only"]
async fn host_routes_a_non_completions_model_through_this_plugin() {
    let host = std::env::var_os("RPI_TEST_CLI").expect("set RPI_TEST_CLI");
    let root = tempfile::tempdir().unwrap();
    let agent = root.path().join("agent");
    std::fs::create_dir(&agent).unwrap();
    let (origin, server) = serve(200, anthropic_reply("routed through opencode-go")).await;
    // Pi's provider base, so every family keeps its own path under the loopback origin.
    std::fs::write(
        agent.join("models.json"),
        json!({"providers":{ID:{"baseUrl":format!("{origin}/inference/go"),"models":[]}}})
            .to_string(),
    )
    .unwrap();
    let mut command = tokio::process::Command::new(host);
    command
        .current_dir(root.path())
        .env("RPI_CODING_AGENT_DIR", &agent)
        .env("OPENCODE_API_KEY", KEY)
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("HTTP_PROXY")
        .env_remove("HTTPS_PROXY")
        .env_remove("ALL_PROXY")
        .arg("--extension")
        .arg(library_path())
        .args([
            "--offline",
            "--package-only",
            "--no-session",
            "--no-builtin-tools",
            "--no-skills",
            "--no-context-files",
            "--no-prompt-templates",
            "--provider",
            ID,
            "--model",
            "claude-haiku-5-5",
            "--print",
            "hello",
        ])
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(20), command.output())
        .await
        .unwrap()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("routed through opencode-go"), "{stdout}");
    assert!(!stdout.contains(KEY) && !stderr.contains(KEY));
    let (head, body) = server.await.unwrap();
    assert!(
        head.starts_with("post /inference/go/v1/messages "),
        "{head}"
    );
    assert_eq!(header_values(&head, "x-api-key"), [KEY]);
    assert_eq!(body["model"], "claude-haiku-5-5");
    // rpi 0.12.0 sends "ephemeral" as the session id under --no-session.
    assert_eq!(header_values(&head, "x-opencode-session"), ["ephemeral"]);
}
