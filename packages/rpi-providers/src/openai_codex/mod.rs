//! OpenAI Codex (ChatGPT Plus/Pro subscription), ported from Pi's built-in `openai-codex`
//! provider (pi-ai 1.3.0 `providers/openai-codex.ts`, `api/openai-codex-responses.ts` and
//! `auth/oauth/openai-codex.ts`; MIT, see `LICENSE` in this directory).
//!
//! Requests go to `<base>/codex/responses` over SSE with Pi's body and headers. The body and the
//! Responses event mapping come from rpi-ai's Responses provider. Pi's WebSocket transport, zstd
//! request compression and device-code login are not ported.
pub mod auth;
mod catalog;

use rpi_ai::{
    provider::set_http_header, providers::openai_responses::OpenAiResponsesProvider,
    AssistantMessageEventStream, CacheRetention, Context, Model, Provider, SimpleStreamOptions,
    ThinkingLevel,
};
use rpi_plugin_sdk::{
    callbacks::free_string, provider::ProviderContext, PluginApi, StbString, StbStringRef,
};
use serde_json::{json, Value};
use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub const PROVIDER_ID: &str = "openai-codex";
/// Pi's default model for this provider (`model-resolver.ts`).
pub const DEFAULT_MODEL_ID: &str = "gpt-6.1-sol";
const BASE_URL: &str = "https://chatgpt.com/backend-api";
/// Pi's `clampOpenAIPromptCacheKey` limit, in characters.
const PROMPT_CACHE_KEY_MAX: usize = 64;

pub fn models() -> Vec<Model> {
    catalog::models()
}

/// Pi's `resolveCodexUrl`.
fn codex_url(base: &str) -> String {
    let base = if base.trim().is_empty() {
        BASE_URL
    } else {
        base
    };
    let base = base.trim_end_matches('/');
    if base.ends_with("/codex/responses") {
        base.into()
    } else if base.ends_with("/codex") {
        format!("{base}/responses")
    } else {
        format!("{base}/codex/responses")
    }
}

/// Pi's `buildRequestBody` reasoning: the clamped level through the model's map, with
/// `summary: "auto"`; with thinking off, the model's "off" value (default `none`) unless null.
fn reasoning(model: &Model, level: Option<ThinkingLevel>) -> Option<Value> {
    let map = model.thinking_level_map.as_ref();
    match level
        .map(|level| model.clamp_thinking_level(level))
        .filter(|level| *level != ThinkingLevel::Off)
    {
        Some(level) => {
            let effort = match map.and_then(|map| map.get(&level)) {
                Some(mapped) => mapped.clone()?,
                None => serde_json::to_value(level).ok()?.as_str()?.to_owned(),
            };
            Some(json!({"effort": effort, "summary": "auto"}))
        }
        None if model.reasoning => match map.and_then(|map| map.get(&ThinkingLevel::Off)) {
            Some(None) => None,
            Some(Some(off)) => Some(json!({"effort": off})),
            None => Some(json!({"effort": "none"})),
        },
        None => None,
    }
}

pub struct OpenAiCodexProvider {
    models: Vec<Model>,
    client: reqwest::Client,
}

#[async_trait::async_trait]
impl Provider for OpenAiCodexProvider {
    fn id(&self) -> &str {
        PROVIDER_ID
    }
    fn models(&self) -> &[Model] {
        &self.models
    }

    async fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let mut model = model.clone();
        model.provider = PROVIDER_ID.into();
        let Some(access) = options.api_key.clone().filter(|key| !key.is_empty()) else {
            return crate::transport::error_stream(
                &model,
                format!("No API key for provider: {PROVIDER_ID}. Run `rpi auth login --provider {PROVIDER_ID}`."),
            );
        };
        let (token, packed) = auth::unpack(&access);
        let Some(account) = packed
            .map(str::to_owned)
            .or_else(|| auth::account_id(token))
        else {
            return crate::transport::error_stream(
                &model,
                "Failed to extract accountId from token".into(),
            );
        };
        // rpi-ai appends `/v1/responses` to a base without `/v1`; the fragment absorbs that
        // suffix and never reaches the wire, so the request path is Pi's.
        model.base_url = format!("{}#/v1", codex_url(&model.base_url));

        // Pi's defaults sit under the model's headers, which sit under the caller's.
        let mut defaults = std::collections::BTreeMap::from([
            ("originator".to_owned(), "pi".to_owned()),
            ("User-Agent".to_owned(), user_agent()),
        ]);
        for (name, value) in model.headers.iter().flatten() {
            set_http_header(&mut defaults, name, Some(value));
        }
        model.headers = Some(defaults);
        let mut options = options.clone();
        let session = (options.cache_retention != CacheRetention::None)
            .then(|| options.session_id.clone())
            .flatten()
            .map(|id| id.chars().take(PROMPT_CACHE_KEY_MAX).collect::<String>());
        let headers = options.headers.get_or_insert_with(Default::default);
        for (name, value) in [
            ("chatgpt-account-id", Some(account.as_str())),
            ("OpenAI-Beta", Some("responses=experimental")),
            ("accept", Some("text/event-stream")),
            ("session-id", session.as_deref()),
            ("x-client-request-id", session.as_deref()),
        ] {
            if value.is_some() {
                set_http_header(headers, name, value);
            }
        }
        options.api_key = Some(token.to_owned());

        // Pi's body fields rpi-ai does not send; sampling params are merged last, so they win.
        let mut body = serde_json::Map::new();
        let instructions = context
            .system_prompt
            .clone()
            .filter(|prompt| !prompt.is_empty());
        body.insert(
            "instructions".into(),
            instructions
                .unwrap_or_else(|| "You are a helpful assistant.".into())
                .into(),
        );
        body.insert("text".into(), json!({"verbosity": "low"}));
        body.insert("include".into(), json!(["reasoning.encrypted_content"]));
        body.insert("tool_choice".into(), "auto".into());
        body.insert("parallel_tool_calls".into(), true.into());
        if let Some(session) = session {
            body.insert("prompt_cache_key".into(), session.into());
        }
        if let Some(temperature) = options.temperature {
            body.insert("temperature".into(), temperature.into());
        }
        if let Some(reasoning) = reasoning(&model, options.reasoning) {
            body.insert("reasoning".into(), reasoning);
        }
        model
            .sampling_params
            .get_or_insert_with(Default::default)
            .extend(body);
        // Pi sends neither of these; rpi-ai would derive `max_output_tokens` and a reasoning
        // field of its own from them.
        options.max_tokens = Some(0);
        options.reasoning = None;

        let stream = OpenAiResponsesProvider::with_models_without_env_api_key(
            PROVIDER_ID,
            None,
            self.client.clone(),
            vec![model.clone()],
        )
        .stream_simple(&model, context, &options)
        .await;
        crate::transport::guard(stream, token.to_owned())
    }
}

/// Pi's `getPiUserAgent`, without the OS release (std has no portable way to read it).
fn user_agent() -> String {
    let platform = match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    };
    format!("pi ({platform}; {arch})")
}

fn create(
    context: &ProviderContext,
    model: &Model,
    _options: &SimpleStreamOptions,
) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(OpenAiCodexProvider {
        models: models(),
        client: context.client(&codex_url(&model.base_url))?,
    }))
}

/// The streaming registration. The macro emits a `register` function, so it gets a module.
mod streaming {
    crate::registration::register_provider!(
        super::PROVIDER_ID,
        super::BASE_URL,
        "openai-responses",
        serde_json::json!({"id": super::PROVIDER_ID, "models": super::models(), "requiresAuth": true}),
        super::create
    );
}

/// Declares the streaming provider and its OAuth flow. Streaming never depends on OAuth.
pub(crate) fn register(api: &PluginApi) -> i32 {
    let result = streaming::register(api);
    if result != 0 {
        return result;
    }
    let Some(register_oauth) = api.register_oauth_provider else {
        return 0;
    };
    let manifest = json!({
        "id": PROVIDER_ID,
        "displayName": "OpenAI (ChatGPT Plus/Pro)",
        "supportsPkce": true,
    })
    .to_string();
    register_oauth(
        StbStringRef::from_str(&manifest),
        oauth_request,
        free_string,
        std::ptr::null_mut::<c_void>(),
    )
}

extern "C" fn oauth_request(
    request: StbStringRef,
    out: *mut StbString,
    _user_data: *mut c_void,
) -> i32 {
    if out.is_null() {
        return 2;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the host keeps the request buffer valid until this callback returns.
        let text = unsafe { request.as_str() }.to_owned();
        oauth_action(&text)
    }));
    let (status, body) = match result {
        Ok(Ok(value)) => (0, value),
        Ok(Err(message)) => (1, error_envelope(&message)),
        Err(_) => (1, error_envelope("OpenAI Codex OAuth panicked")),
    };
    // SAFETY: `out` is the host's exclusive output slot, checked non-null above.
    unsafe { *out = StbString::from_string(body.to_string()) };
    status
}

fn error_envelope(message: &str) -> Value {
    json!({"version": 1, "kind": "error", "payload": {"message": message}})
}

/// Issued logins: (state, verifier, issued at). Single use, ten minutes, at most eight.
fn logins() -> &'static Mutex<Vec<(String, String, Instant)>> {
    static LOGINS: Mutex<Vec<(String, String, Instant)>> = Mutex::new(Vec::new());
    &LOGINS
}
const LOGIN_TTL: Duration = Duration::from_secs(10 * 60);

fn oauth_action(text: &str) -> Result<Value, String> {
    let request: Value =
        serde_json::from_str(text).map_err(|_| "invalid OAuth request".to_owned())?;
    let input = &request["input"];
    match request["action"].as_str() {
        Some("begin") => begin(),
        Some("exchange") => exchange(input["value"].as_str().unwrap_or_default()),
        Some("refresh") => {
            let refresh = input["credential"]["refresh"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            if refresh.is_empty() {
                return Err("OpenAI Codex has no stored refresh credential".into());
            }
            block_on(move |client| async move {
                auth::refresh(&client, auth::TOKEN_URL, &refresh).await
            })
        }
        // Pi has no revocation for this provider; the host still removes the stored credential.
        Some("revoke") => Ok(json!({"version": 1, "kind": "success", "payload": {}})),
        _ => Err("OAuth action must be one of begin, exchange, refresh, revoke".into()),
    }
}

fn begin() -> Result<Value, String> {
    let auth::Login {
        url,
        state,
        verifier,
    } = auth::authorize()?;
    let mut logins = logins().lock().unwrap();
    logins.retain(|(.., at)| at.elapsed() < LOGIN_TTL);
    if logins.len() >= 8 {
        logins.remove(0);
    }
    logins.push((state, verifier, Instant::now()));
    Ok(json!({
        "version": 1,
        "kind": "interaction",
        "payload": {"display": {
            "title": "Connect ChatGPT (OpenAI Codex)",
            "url": url,
            "instructions": [
                "Open the URL and sign in with your ChatGPT account.",
                format!("The browser then opens {}, which does not load; paste that full URL (or the code) here.", auth::REDIRECT_URI),
            ],
        }},
    }))
}

/// Pi's manual-input path: a state, when present, must match; a bare code redeems the latest
/// login. A rejected input sends nothing.
fn exchange(pasted: &str) -> Result<Value, String> {
    let (code, state) = auth::parse_input(pasted);
    let verifier = {
        let mut logins = logins().lock().unwrap();
        logins.retain(|(.., at)| at.elapsed() < LOGIN_TTL);
        let index = match &state {
            Some(state) => logins.iter().position(|(issued, ..)| issued == state),
            None => logins.len().checked_sub(1),
        };
        let Some(index) = index else {
            return Err(if state.is_some() {
                "State mismatch".into()
            } else {
                "Start OpenAI Codex login before pasting the code".into()
            });
        };
        if code.as_deref().unwrap_or_default().is_empty() {
            return Err("Missing authorization code".into());
        }
        logins.remove(index).1
    };
    let code = code.unwrap_or_default();
    block_on(move |client| async move {
        auth::exchange(&client, auth::TOKEN_URL, &code, &verifier).await
    })
}

/// Runs one token request on its own thread and runtime, so a host runtime is never nested.
fn block_on<F, Fut>(work: F) -> Result<Value, String>
where
    F: FnOnce(reqwest::Client) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<Value, String>>,
{
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        runtime.block_on(async move {
            work(rpi_ai::http::build_client(auth::TOKEN_URL, None, None)?).await
        })
    })
    .join()
    .map_err(|_| "OpenAI Codex OAuth worker stopped".to_owned())?
}
