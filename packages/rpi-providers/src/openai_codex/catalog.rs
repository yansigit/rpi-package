//! `catalog.json` is pi-ai 1.3.0 `providers/data/openai-codex.json`, unchanged.
use rpi_ai::{Api, Model, StreamingProtocolCompat};
use serde_json::{json, Value};

/// Pi's chat models in Pi's order, Pi's default first. Pi's `openai-codex-responses` models are
/// declared as `openai-responses`: rpi-ai streams them with its Responses protocol, and its
/// reasoning replay only matches messages whose API equals the model's.
pub(crate) fn models() -> Vec<Model> {
    let groups: Value = serde_json::from_str(include_str!("catalog.json")).expect("pinned catalog");
    let mut models: Vec<Model> = groups
        .as_object()
        .into_iter()
        .flat_map(|groups| groups.values())
        .filter_map(Value::as_object)
        .flat_map(|group| group.values().cloned())
        .filter(|entry| entry["type"] == "chat")
        .map(|mut entry| {
            let compat = entry
                .as_object_mut()
                .and_then(|entry| entry.remove("compat"))
                .unwrap_or_else(|| json!({}));
            let mut model: Model = serde_json::from_value(entry).expect("pinned model");
            model.api = Api::OpenaiResponses;
            model.compat = Some(StreamingProtocolCompat::OpenaiResponses(
                serde_json::from_value(compat).expect("pinned compat"),
            ));
            model.cost_configured = true;
            model
        })
        .collect();
    models.sort_by_key(|model| model.id != super::DEFAULT_MODEL_ID);
    models
}
