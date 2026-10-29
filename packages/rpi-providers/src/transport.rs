//! Shared transport without retaining request credentials in the client pool.
use rpi_ai::{
    event_stream::create_assistant_message_event_stream,
    providers::openai_completions::OpenAiCompletionsProvider, AssistantMessage,
    AssistantMessageEvent, AssistantMessageEventStream, Context, Model, Provider,
    SimpleStreamOptions, StopReason,
};
use std::sync::Arc;

pub(crate) async fn stream(
    client: Option<&reqwest::Client>,
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    anonymous: bool,
) -> AssistantMessageEventStream {
    let protocol: Result<Arc<dyn Provider>, String> = match client {
        Some(client) if anonymous && options.api_key.is_none() => Ok(Arc::new(
            OpenAiCompletionsProvider::with_models_without_auth(
                model.provider.clone(),
                client.clone(),
                vec![model.clone()],
            ),
        )),
        Some(client) => Ok(Arc::new(
            OpenAiCompletionsProvider::with_models_without_env_api_key(
                model.provider.clone(),
                options.api_key.clone(),
                client.clone(),
                vec![model.clone()],
            ),
        )),
        None => rpi_plugin_sdk::provider::completions(model, options, anonymous),
    };
    match protocol {
        Ok(provider) => provider.stream_simple(model, context, options).await,
        Err(error) => error_stream(model, error),
    }
}

/// A stream holding only an error terminal, for failures before any request is sent.
pub(crate) fn error_stream(model: &Model, error: String) -> AssistantMessageEventStream {
    let (mut producer, stream) = create_assistant_message_event_stream();
    producer.push(AssistantMessageEvent::Error {
        reason: rpi_ai::types::ErrorReason::Error,
        error: AssistantMessage::terminal(
            model.api.clone(),
            model.provider.clone(),
            model.id.clone(),
            StopReason::Error,
            error,
            0,
        ),
    });
    stream
}

/// Gateways may echo a rejected credential in the error body; the terminal never carries it.
/// A rate-limited terminal gains the shared `rate_limited;` marker (see `rate_limit`).
pub(crate) fn guard(
    mut inner: AssistantMessageEventStream,
    secret: String,
) -> AssistantMessageEventStream {
    let (mut producer, stream) = create_assistant_message_event_stream();
    tokio::spawn(async move {
        while let Some(mut event) = inner.next().await {
            if let AssistantMessageEvent::Error { reason, error } = &mut event {
                if let Some(text) = error.error_message.as_mut() {
                    if !secret.is_empty() {
                        *text = text.replace(&secret, "[credential]");
                    }
                    if *reason != rpi_ai::ErrorReason::Aborted {
                        *text = crate::rate_limit::mark(text);
                    }
                }
            }
            producer.push(event);
        }
    });
    stream
}
