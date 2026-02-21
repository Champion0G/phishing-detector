import re
import math
from urllib.parse import urlparse

# Suspicious keywords commonly found in phishing URLs
SUSPICIOUS_WORDS = [
    "login", "secure", "verify", "update", "account", "bank",
    "confirm", "password", "paypal", "signin", "webscr", "ebay",
    "support", "billing", "wallet", "free", "lucky", "winner"
]

# TLDs frequently abused in phishing campaigns
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "pw",
    "cc", "men", "work", "click", "link", "online", "site",
    "live", "stream", "download", "loan", "win"
}


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def _max_consecutive_digits(s: str) -> int:
    """Return the length of the longest run of consecutive digits."""
    max_run = cur = 0
    for c in s:
        if c.isdigit():
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def extract_features(url: str) -> dict:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # ── Basic counts ──────────────────────────────────────────────────────────
    features = {}
    features["url_length"]        = len(url)
    features["hostname_length"]   = len(hostname)
    features["num_digits"]        = sum(c.isdigit() for c in url)
    features["num_dots"]          = url.count(".")
    features["num_hyphens"]       = url.count("-")
    features["num_special_chars"] = len(re.findall(r'[!@#$%^&*(),?":{}|<>]', url))

    # ── IP address check ──────────────────────────────────────────────────────
    features["has_ip"] = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", hostname)))

    # ── Suspicious word count ─────────────────────────────────────────────────
    url_lower = url.lower()
    features["num_suspicious_words"] = sum(w in url_lower for w in SUSPICIOUS_WORDS)

    # ── Entropy ───────────────────────────────────────────────────────────────
    features["domain_entropy"] = _shannon_entropy(hostname)
    features["url_entropy"]    = _shannon_entropy(url)

    # ── Digit / letter ratio ──────────────────────────────────────────────────
    letters = sum(c.isalpha() for c in url)
    digits  = sum(c.isdigit() for c in url)
    features["digit_to_letter_ratio"] = digits / (letters + 1)  # +1 avoids /0

    # ── TLD features ──────────────────────────────────────────────────────────
    parts = hostname.split(".")
    tld   = parts[-1].lower() if parts else ""
    features["tld_length"]       = len(tld)
    features["is_suspicious_tld"] = int(tld in SUSPICIOUS_TLDS)

    # ── Subdomain depth ───────────────────────────────────────────────────────
    # e.g. "a.b.example.com" → 2 subdomains
    features["subdomain_count"] = max(0, len(parts) - 2)

    # ── Path features ─────────────────────────────────────────────────────────
    features["path_length"] = len(path)
    features["path_depth"]  = path.count("/")

    # ── Query string ──────────────────────────────────────────────────────────
    features["has_query"]    = int(bool(query))
    features["query_length"] = len(query)

    # ── Special phishing signals ──────────────────────────────────────────────
    features["has_at_symbol"]  = int("@" in url)
    features["has_double_slash"] = int("//" in path)
    features["num_equals"]     = url.count("=")
    features["num_ampersand"]  = url.count("&")
    features["num_percent"]    = url.count("%")

    # ── Vowel ratio in hostname ───────────────────────────────────────────────
    vowels = sum(c in "aeiou" for c in hostname.lower())
    features["vowel_ratio"] = vowels / (len(hostname) + 1)

    # ── Max consecutive digits ────────────────────────────────────────────────
    features["max_consecutive_digits"] = _max_consecutive_digits(url)

    return features