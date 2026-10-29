//! Pi's ChatGPT OAuth (`auth/oauth/openai-codex.js` browser flow): a public PKCE client, so no
//! secret ships. The host stores the credential; this module owns the wire protocol.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};
use url::Url;

pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const SCOPE: &str = "openid profile email offline_access";
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

/// A login URL, the state its redirect must return, and the PKCE verifier that redeems it.
pub struct Login {
    pub url: String,
    pub state: String,
    pub verifier: String,
}

/// Pi's `createAuthorizationFlow` with the default originator.
pub fn authorize() -> Result<Login, String> {
    let mut verifier = [0; 32];
    let mut state = [0; 16];
    getrandom::getrandom(&mut verifier).map_err(|error| error.to_string())?;
    getrandom::getrandom(&mut state).map_err(|error| error.to_string())?;
    let verifier = URL_SAFE_NO_PAD.encode(verifier);
    let state: String = state.iter().map(|byte| format!("{byte:02x}")).collect();
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    let mut url = Url::parse(AUTHORIZE_URL).expect("pinned URL");
    url.query_pairs_mut().extend_pairs([
        ("response_type", "code"),
        ("client_id", CLIENT_ID),
        ("redirect_uri", REDIRECT_URI),
        ("scope", SCOPE),
        ("code_challenge", &challenge),
        ("code_challenge_method", "S256"),
        ("state", &state),
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("originator", "pi"),
    ]);
    Ok(Login {
        url: url.into(),
        state,
        verifier,
    })
}

/// Pi's `parseAuthorizationInput`: a redirect URL, `code#state`, a query string, or a bare code.
pub fn parse_input(input: &str) -> (Option<String>, Option<String>) {
    let value = input.trim();
    if value.is_empty() {
        return (None, None);
    }
    let pairs = |query: &str| {
        let get = |name: &str| {
            url::form_urlencoded::parse(query.as_bytes())
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.into_owned())
        };
        (get("code"), get("state"))
    };
    if let Ok(url) = Url::parse(value) {
        return pairs(url.query().unwrap_or_default());
    }
    if let Some((code, state)) = value.split_once('#') {
        return (Some(code.into()), Some(state.into()));
    }
    if value.contains("code=") {
        return pairs(value);
    }
    (Some(value.into()), None)
}

/// The ChatGPT account claim of an access token (Pi's `getAccountId`).
pub fn account_id(token: &str) -> Option<String> {
    let mut parts = token.split('.');
    let (Some(_), Some(payload), Some(_), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return None;
    };
    let payload = URL_SAFE_NO_PAD.decode(payload.trim_end_matches('=')).ok()?;
    let payload: Value = serde_json::from_slice(&payload).ok()?;
    payload[JWT_CLAIM_PATH]["chatgpt_account_id"]
        .as_str()
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
}

/// The stored access value is `<token>|<account>`, as Antigravity stores its project.
pub fn unpack(access: &str) -> (&str, Option<&str>) {
    match access.split_once('|') {
        Some((token, account)) => (token, Some(account).filter(|id| !id.is_empty())),
        None => (access, None),
    }
}

/// Redeems an authorization code at `token_url` and returns the host credential envelope.
pub async fn exchange(
    client: &reqwest::Client,
    token_url: &str,
    code: &str,
    verifier: &str,
) -> Result<Value, String> {
    let form = [
        ("grant_type", "authorization_code"),
        ("client_id", CLIENT_ID),
        ("code", code),
        ("code_verifier", verifier),
        ("redirect_uri", REDIRECT_URI),
    ];
    token(client, token_url, "exchange", &form, &[code, verifier]).await
}

/// Refreshes at `token_url` and returns the host credential envelope.
pub async fn refresh(
    client: &reqwest::Client,
    token_url: &str,
    refresh: &str,
) -> Result<Value, String> {
    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh),
        ("client_id", CLIENT_ID),
    ];
    token(client, token_url, "refresh", &form, &[refresh]).await
}

/// Pi's `readTokenResponse` and `credentialsFromToken`. Error text never carries the request's
/// secrets or the issued tokens (Pi prints the whole response when a field is missing).
async fn token(
    client: &reqwest::Client,
    token_url: &str,
    operation: &str,
    form: &[(&str, &str)],
    secrets: &[&str],
) -> Result<Value, String> {
    let response = client
        .post(token_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(
            url::form_urlencoded::Serializer::new(String::new())
                .extend_pairs(form)
                .finish(),
        )
        .send()
        .await
        .map_err(|error| {
            format!(
                "OpenAI Codex token {operation} error: {}",
                error.without_url()
            )
        })?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        let text = secrets
            .iter()
            .filter(|secret| !secret.is_empty())
            .fold(text, |text, secret| text.replace(secret, "[credential]"));
        return Err(format!(
            "OpenAI Codex token {operation} failed ({}): {text}",
            status.as_u16()
        ));
    }
    let tokens: Value = serde_json::from_str(&text).unwrap_or_default();
    let (Some(access), Some(refresh), Some(expires_in)) = (
        tokens["access_token"]
            .as_str()
            .filter(|value| !value.is_empty()),
        tokens["refresh_token"]
            .as_str()
            .filter(|value| !value.is_empty()),
        tokens["expires_in"].as_f64(),
    ) else {
        return Err(format!(
            "OpenAI Codex token {operation} response missing fields"
        ));
    };
    let account = account_id(access).ok_or("Failed to extract accountId from token")?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs() as i64);
    Ok(json!({
        "version": 1,
        "kind": "credential",
        "payload": {
            "access": format!("{access}|{account}"),
            "refresh": refresh,
            // Host expiry is whole seconds; Pi's is milliseconds.
            "expires_at": now + expires_in as i64,
        },
    }))
}
