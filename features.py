
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://(\d{1,3}\.){3}\d{1,3}")
SUSPICIOUS_TLDS = (".tk", ".xyz", ".click", ".info", ".biz", ".top", ".gq", ".cf", ".ml")

URGENCY_WORDS = [
    "urgent", "immediately", "immediate action", "act now", "asap",
    "as soon as possible", "expire", "expires", "expiring", "24 hours",
    "final notice", "limited time", "act fast", "right away",
    "before it expires", "time sensitive", "suspend", "suspended",
    "suspension", "locked", "restricted",
]

ACTION_WORDS = [
    "click here", "click below", "verify", "confirm", "update your",
    "log in", "login", "sign in", "reset your password",
    "provide your", "confirm your identity", "validate your account",
]

SENSITIVE_INFO_WORDS = [
    "password", "ssn", "social security", "bank account", "account number",
    "credit card", "pin number", "routing number", "security code",
    "card number",
]

GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear valued customer", "dear member",
    "dear account holder", "dear sir/madam", "dear client",
]


def _count_occurrences(text, phrase_list):
    text_lower = text.lower()
    return sum(text_lower.count(p) for p in phrase_list)


def _has_any(text, phrase_list):
    text_lower = text.lower()
    return int(any(p in text_lower for p in phrase_list))


def extract_handcrafted_features(text):
    """
    Given raw (uncleaned, original-case) email text, return a dict of
    hand-crafted numeric features.
    """
    text = text or ""
    urls = URL_RE.findall(text)
    num_urls = len(urls)
    num_ip_urls = len(IP_URL_RE.findall(text))
    num_suspicious_tld = sum(
        1 for url in urls if any(tld in url.lower() for tld in SUSPICIOUS_TLDS)
    )
    num_hyphen_domains = sum(1 for url in urls if url.count("-") >= 2)

    exclamations = text.count("!")
    length = len(text)
    upper_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = (upper_chars / length) if length > 0 else 0.0

    features = {
        "num_urls": num_urls,
        "num_ip_urls": num_ip_urls,
        "num_suspicious_tld_urls": num_suspicious_tld,
        "num_hyphenated_domains": num_hyphen_domains,
        "urgency_word_count": _count_occurrences(text, URGENCY_WORDS),
        "has_urgency_language": _has_any(text, URGENCY_WORDS),
        "action_word_count": _count_occurrences(text, ACTION_WORDS),
        "has_action_words": _has_any(text, ACTION_WORDS),
        "sensitive_info_request": _has_any(text, SENSITIVE_INFO_WORDS),
        "generic_greeting": _has_any(text, GENERIC_GREETINGS),
        "exclamation_count": exclamations,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "text_length": length,
    }
    return features


HANDCRAFTED_FEATURE_NAMES = [
    "num_urls", "num_ip_urls", "num_suspicious_tld_urls",
    "num_hyphenated_domains", "urgency_word_count", "has_urgency_language",
    "action_word_count", "has_action_words", "sensitive_info_request",
    "generic_greeting", "exclamation_count", "uppercase_ratio", "text_length",
]


def handcrafted_features_to_vector(feature_dict):
    """Convert a feature dict into a fixed-order numeric numpy vector."""
    return np.array([feature_dict[name] for name in HANDCRAFTED_FEATURE_NAMES], dtype=float)


def build_tfidf_vectorizer(max_features=3000):
    """
    Build the TF-IDF vectorizer used for text features.
    Uses unigrams + bigrams so short phishing phrases like
    "click here" or "verify your account" are captured.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
    )
