//! OpenCode Go, ported from Pi's built-in `opencode-go` provider (pi-ai 1.3.0
//! `providers/opencode-go.ts` and `opencode-headers.ts`; MIT, see `LICENSE` in this directory).
//!
//! One key and one catalog whose models speak different wire APIs. Each request is
//! dispatched to the rpi-ai protocol for the model's Pi API and carries Pi's
//! `x-opencode-session` routing header.
mod catalog;

use rpi_ai::{
    event_stream::create_assistant_message_event_stream,
    providers::{anthropic::AnthropicProvider, openai_responses::OpenAiResponsesProvider},
    Api, AssistantMessageEvent, AssistantMessageEventStream, Context, Model, Provider,
    SimpleStreamOptions,
};
use serde_json::json;
use std::sync::Arc;

pub const PROVIDER_ID: &str = "opencode-go";
/// Pi's default model for this provider (`model-resolver.ts`).
pub const DEFAULT_MODEL_ID: &str = "kimi-k3";
const BASE_URL: &str = "https://opencode.ai/inference/go";
const API_KEY_ENV: &str = "OPENCODE_API_KEY";
const SESSION_HEADER: &str = "x-opencode-session";

pub fn models() -> Vec<Model> {
    catalog::models()
}

/// Pi's `withSessionHeader`: add the session id unless the caller already set the header.
fn with_session_header(options: &mut SimpleStreamOptions) {
    let Some(session) = options.session_id.clone().filter(|id| !id.is_empty()) else {
        return;
    };
    let headers = options.headers.get_or_insert_with(Default::default);
    if !headers
        .keys()
        .any(|name| name.eq_ignore_ascii_case(SESSION_HEADER))
    {
        headers.insert(SESSION_HEADER.into(), session);
    }
}

pub struct OpenCodeGoProvider {
    api_key: Option<String>,
    models: Vec<Model>,
    client: Option<reqwest::Client>,
}

impl OpenCodeGoProvider {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            models: models(),
            client: None,
        }
    }
}

#[async_trait::async_trait]
impl Provider for OpenCodeGoProvider {
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
        if let Some(api) = catalog::api(&model.id) {
            model.api = api;
        }
        if model.base_url.is_empty() {
            model.base_url = BASE_URL.into();
        }
        let mut options = options.clone();
        if options.api_key.is_none() {
            options.api_key = self.api_key.clone();
        }
        if options.api_key.is_none() {
            return crate::transport::error_stream(
                &model,
                format!(
                    "No API key for {PROVIDER_ID}. Set {API_KEY_ENV} or run `rpi auth login --provider {PROVIDER_ID}`."
                ),
            );
        }
        with_session_header(&mut options);
        let client = match self.client.clone().map_or_else(
            || rpi_ai::http::build_client(&model.base_url, None, None),
            Ok,
        ) {
            Ok(client) => client,
            Err(error) => return crate::transport::error_stream(&model, error),
        };
        // The key travels in options; the protocol providers never read another key.
        let models = vec![model.clone()];
        let stream = match model.api {
            Api::AnthropicMessages => {
                AnthropicProvider::with_models_without_env_api_key(None, client, models)
                    .stream_simple(&model, context, &options)
                    .await
            }
            Api::OpenaiResponses => {
                OpenAiResponsesProvider::with_models_without_env_api_key(
                    PROVIDER_ID,
                    None,
                    client,
                    models,
                )
                .stream_simple(&model, context, &options)
                .await
            }
            _ => crate::transport::stream(Some(&client), &model, context, &options, false).await,
        };
        redact(stream, options.api_key.unwrap_or_default())
    }
}

/// Gateways may echo a rejected key in the error body; the terminal never carries it.
fn redact(mut inner: AssistantMessageEventStream, key: String) -> AssistantMessageEventStream {
    let (mut producer, stream) = create_assistant_message_event_stream();
    tokio::spawn(async move {
        while let Some(mut event) = inner.next().await {
            if let AssistantMessageEvent::Error { error, .. } = &mut event {
                if let Some(text) = error.error_message.as_mut().filter(|_| !key.is_empty()) {
                    *text = text.replace(&key, "[credential]");
                }
            }
            producer.push(event);
        }
    });
    stream
}

fn create(
    context: &rpi_plugin_sdk::provider::ProviderContext,
    model: &Model,
    options: &SimpleStreamOptions,
) -> Result<Arc<dyn Provider>, String> {
    let base = if model.base_url.is_empty() {
        BASE_URL
    } else {
        &model.base_url
    };
    Ok(Arc::new(OpenCodeGoProvider {
        api_key: options.api_key.clone(),
        models: models(),
        client: Some(context.client(base)?),
    }))
}

crate::registration::register_provider!(
    PROVIDER_ID,
    BASE_URL,
    "openai-completions",
    json!({
        "id": PROVIDER_ID, "models": models(),
        "apiKeyEnv": API_KEY_ENV, "requiresAuth": true
    }),
    create
);
