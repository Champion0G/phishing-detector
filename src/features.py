import re
from urllib.parse import urlparse

suspicious_words = ["login", "secure", "verify", "update", "account", "bank"]

def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname if parsed.hostname else ""

    features = {}

    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname)
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_special_chars"] = len(re.findall(r"[!@#$%^&*(),?\":{}|<>]", url))
    features["has_ip"] = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", hostname)))
    features["num_suspicious_words"] = sum(word in url.lower() for word in suspicious_words)

    return features