//! The rate-limit marker shared with account pools (Zephyr parses it): a rate-limited error
//! terminal's text starts with `rate_limited; resets_at=<unix seconds> ` when the reset time is
//! known, otherwise with `rate_limited; `, followed by the original (redacted) message.
use regex::Regex;
use std::{
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

fn expression(storage: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    storage.get_or_init(|| Regex::new(pattern).expect("pinned rate-limit expression"))
}

/// Prefixes the marker when `message` reports a rate limit; any other message is returned as is.
pub(crate) fn mark(message: &str) -> String {
    static LIMITED: OnceLock<Regex> = OnceLock::new();
    // `http error 429` is rpi-ai's HTTP failure text, `HTTP 429` and `code 429` Antigravity's.
    let limited = expression(
        &LIMITED,
        r"(?i)\bhttp(?: error)? 429\b|\bcode 429\b|usage_limit_reached|rate.?limit|resource_exhausted|too many requests",
    );
    if message.starts_with("rate_limited;") || !limited.is_match(message) {
        return message.into();
    }
    match resets_at(message, now()) {
        Some(at) => format!("rate_limited; resets_at={at} {message}"),
        None => format!("rate_limited; {message}"),
    }
}

/// The reset time a provider body reports: Codex's `resets_at` (unix seconds) or
/// `resets_in_seconds`, or Google's RetryInfo `retryDelay` (`"30s"`).
// shortcut: response headers (retry-after, x-codex-*-reset) are not read, because rpi-ai 0.8.3
// drops them before the error text; read them if a provider gets its own transport.
fn resets_at(message: &str, now: i64) -> Option<i64> {
    static AT: OnceLock<Regex> = OnceLock::new();
    static IN: OnceLock<Regex> = OnceLock::new();
    static DELAY: OnceLock<Regex> = OnceLock::new();
    let number = |regex: &Regex| {
        regex
            .captures(message)
            .and_then(|captures| captures[1].parse::<f64>().ok())
    };
    number(expression(&AT, r#""resets_at"\s*:\s*(\d+)"#))
        .map(|at| at as i64)
        .or_else(|| {
            number(expression(&IN, r#""resets_in_seconds"\s*:\s*(\d+)"#))
                .or_else(|| {
                    number(expression(
                        &DELAY,
                        r#""retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s""#,
                    ))
                })
                .map(|seconds| now + seconds.ceil() as i64)
        })
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_rate_limits_and_leaves_other_errors() {
        assert_eq!(mark("http error 401: bad key"), "http error 401: bad key");
        assert_eq!(mark("http error 500: boom"), "http error 500: boom");
        assert_eq!(
            mark("http error 429: slow down"),
            "rate_limited; http error 429: slow down"
        );
        let codex =
            r#"http error 429: {"error":{"type":"usage_limit_reached","resets_at":1893456000}}"#;
        assert_eq!(
            mark(codex),
            format!("rate_limited; resets_at=1893456000 {codex}")
        );
        // Already marked text is not marked twice.
        assert_eq!(mark(&mark(codex)), mark(codex));
        assert_eq!(
            mark("rate_limit_exceeded: try later"),
            "rate_limited; rate_limit_exceeded: try later"
        );
    }

    #[test]
    fn relative_resets_are_converted_to_unix_seconds() {
        assert_eq!(resets_at(r#"{"resets_in_seconds": 120}"#, 1000), Some(1120));
        assert_eq!(resets_at(r#"{"retryDelay": "1.5s"}"#, 1000), Some(1002));
        assert_eq!(resets_at("no reset here", 1000), None);
    }
}
