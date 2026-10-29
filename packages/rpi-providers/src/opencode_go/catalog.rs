//! `catalog.json` is pi-ai 1.3.0 `providers/data/opencode-go.json`, flattened in Pi's order.
use rpi_ai::{Api, Model, StreamingProtocolCompat};
use serde_json::{json, Value};
use std::sync::OnceLock;

/// Pi's chat models with their own wire API and compat. Models on an API rpi-ai 0.8.3 cannot
/// stream (google-generative-ai) are dropped.
fn all() -> &'static [Model] {
    static MODELS: OnceLock<Vec<Model>> = OnceLock::new();
    MODELS.get_or_init(|| {
        let entries: Vec<Value> =
            serde_json::from_str(include_str!("catalog.json")).expect("pinned catalog");
        entries
            .into_iter()
            .filter_map(|mut entry| {
                // Pi's compat is flat; rpi's is tagged by API, so it is decoded per model.
                let compat = entry
                    .as_object_mut()
                    .and_then(|entry| entry.remove("compat"))
                    .unwrap_or_else(|| json!({}));
                let mut model: Model = serde_json::from_value(entry).expect("pinned model");
                let compat = match model.api {
                    Api::AnthropicMessages => {
                        StreamingProtocolCompat::AnthropicMessages(decode(compat))
                    }
                    Api::OpenaiCompletions => {
                        StreamingProtocolCompat::OpenaiCompletions(decode(compat))
                    }
                    Api::OpenaiResponses => {
                        StreamingProtocolCompat::OpenaiResponses(decode(compat))
                    }
                    _ => return None,
                };
                model.compat = Some(compat);
                model.cost_configured = true;
                Some(model)
            })
            .collect()
    })
}

/// The wire API Pi uses for a catalog model.
pub(crate) fn api(id: &str) -> Option<Api> {
    all()
        .iter()
        .find(|model| model.id == id)
        .map(|model| model.api.clone())
}

/// Registration catalog, Pi's default first. The host routes a model to a plugin by
/// (provider, api) and this plugin registers one style, so every model declares it;
/// `api` above restores the real one per request.
pub(crate) fn models() -> Vec<Model> {
    let mut models = all().to_vec();
    models.sort_by_key(|model| model.id != super::DEFAULT_MODEL_ID);
    for model in &mut models {
        model.api = Api::OpenaiCompletions;
    }
    models
}

fn decode<T: serde::de::DeserializeOwned>(compat: Value) -> T {
    serde_json::from_value(compat).expect("pinned compat")
}
