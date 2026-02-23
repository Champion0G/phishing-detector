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

# TLDs common in professional/tech sectors (to mitigate bias)
PROFESSIONAL_TLDS = {"io", "app", "dev"}

# Domains that are highly unlikely to be phishing (Reputation List)
REPUTABLE_DOMAINS = {
    "google.com", "github.com", "coursera.org", "microsoft.com",
    "apple.com", "amazon.com", "netflix.com", "linkedin.com",
    "facebook.com", "twitter.com", "instagram.com", "youtube.com",
    "paypal.com", "vercel.app", "netlify.app", "ghost.io",
    "github.io", "codedex.io", "herokuap.com", "cloudfront.net",
    "digitalocean.com", "railway.app", "render.com", "bitbucket.org",
    "gitlab.com", "stackoverflow.com", "medium.com"
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


def _get_hostname(url: str) -> str:
    """Extract hostname reliably even if protocol is missing."""
    try:
        if not url.startswith(("http://", "https://", "ftp://")):
            url = "http://" + url
        parsed = urlparse(url)
        return parsed.hostname or ""
    except Exception:
        return ""


def extract_features(url: str) -> dict:
    try:
        hostname = _get_hostname(url)
        parsed = urlparse(url if url.startswith(("http://", "https://")) else "http://" + url)
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        hostname = ""
        path = ""
        query = ""

    # ── Basic counts ──────────────────────────────────────────────────────────
    features = {}
    features["url_length"]        = min(100, len(url))
    features["hostname_length"]   = min(50, len(hostname))
    features["num_digits"]        = sum(c.isdigit() for c in url)
    features["num_dots"]          = url.count(".")
    features["num_hyphens"]       = url.count("-")
    features["num_special_chars"] = len(re.findall(r'[!@#$%^&*(),?":{}|<>]', url))

    # ── IP address check ──────────────────────────────────────────────────────
    features["has_ip"] = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", hostname)))

    # ── Reputation List ───────────────────────────────────────────────────────
    features["is_reputable"] = int(any(domain in hostname.lower() for domain in REPUTABLE_DOMAINS))

    # ── Suspicious word count ─────────────────────────────────────────────────
    url_lower = url.lower()
    features["num_suspicious_words"] = sum(w in url_lower for w in SUSPICIOUS_WORDS)

    # ── Entropy ───────────────────────────────────────────────────────────────
    features["domain_entropy"] = _shannon_entropy(hostname)
    features["url_entropy"]    = _shannon_entropy(url)

    # ── Digit / letter ratio ──────────────────────────────────────────────────
    letters = sum(c.isalpha() for c in url)
    digits  = sum(c.isdigit() for c in url)
    features["digit_to_letter_ratio"] = digits / (letters + 1)

    # ── TLD features ──────────────────────────────────────────────────────────
    parts = hostname.split(".")
    tld   = parts[-1].lower() if parts else ""
    features["tld_length"]       = len(tld)
    features["is_suspicious_tld"] = int(tld in SUSPICIOUS_TLDS)
    features["is_professional_tld"] = int(tld in PROFESSIONAL_TLDS)

    # ── Subdomain depth ───────────────────────────────────────────────────────
    features["subdomain_count"] = max(0, len(parts) - 2)

    # ── Path features ─────────────────────────────────────────────────────────
    features["path_length"] = min(50, len(path))
    features["path_depth"]  = min(3, path.count("/"))

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

    # ── Typosquatting signals ─────────────────────────────────────────────────
    h_digits = sum(c.isdigit() for c in hostname)
    features["digit_density_in_hostname"] = h_digits / (len(hostname) + 1)
    features["char_substitution_risk"] = int(bool(re.search(r"[01][a-z]|[a-z][01]", hostname)))
    features["brand_in_subdomain"] = int(any(domain in hostname.lower() and not hostname.lower().endswith(domain) for domain in REPUTABLE_DOMAINS))

    # ── Max consecutive digits ────────────────────────────────────────────────
    features["max_consecutive_digits"] = _max_consecutive_digits(url)

    # ── High Trust Anchor & Signal Amplification ─────────────────────────────
    # If it's a reputable domain, we suppress structural risks and amplify trust
    if features["is_reputable"]:
        features["path_length"] = 0
        features["path_depth"] = 0
        features["url_length"] = min(20, features["url_length"])
        features["char_substitution_risk"] = 0
        features["digit_density_in_hostname"] = 0
        features["is_reputable"] = 10.0  # Amplify the trust signal
    
    # Soft Trust for professional TLDs if not reputable but clean
    elif features["is_professional_tld"] and features["num_suspicious_words"] == 0:
        features["is_professional_tld"] = 5.0 # Give a trust boost to tech TLDs
    
    # Amplify the typosquatting signal for untrusted domains
    if features["char_substitution_risk"] > 0 and not features["is_reputable"]:
        features["char_substitution_risk"] = 10.0

    return features