

import re
import email
from email import policy
from pathlib import Path

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def load_email_from_file(file_path):
    """
    Load an email from disk. Supports .eml (RFC 822) files and plain
    .txt files. Returns a dict: {"subject": str, "body": str}.
    """
    file_path = Path(file_path)
    raw = file_path.read_text(encoding="utf-8", errors="ignore")

    if file_path.suffix.lower() == ".eml":
        msg = email.message_from_string(raw, policy=policy.default)
        subject = msg.get("subject", "") or ""
        body = _extract_body_from_message(msg)
        return {"subject": subject, "body": body}

    # Plain text file: treat first line starting with "Subject:" as the
    # subject if present, otherwise no subject, whole file is body.
    lines = raw.splitlines()
    subject = ""
    body_start = 0
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":", 1)[1].strip()
        body_start = 1
    body = "\n".join(lines[body_start:])
    return {"subject": subject, "body": body}


def _extract_body_from_message(msg):
    """Walk a parsed email.message.Message and pull out plain text."""
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    parts.append(part.get_content())
                except Exception:
                    pass
            elif content_type == "text/html" and not parts:
                try:
                    parts.append(strip_html(part.get_content()))
                except Exception:
                    pass
        return "\n".join(parts)
    else:
        try:
            content = msg.get_content()
        except Exception:
            content = msg.get_payload()
        if msg.get_content_type() == "text/html":
            content = strip_html(content)
        return content


def strip_html(html_text):
    """Remove HTML tags, collapsing whitespace left behind."""
    text = HTML_TAG_RE.sub(" ", html_text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_text(text):
    """
    Normalise text for vectorization:
        - strip HTML tags
        - lower-case
        - collapse repeated whitespace
    Keeps letters/numbers/punctuation intact (TF-IDF handles the rest).
    """
    if not text:
        return ""
    text = strip_html(text)
    text = text.lower()
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def combine_subject_body(subject, body):
    """Combine subject and body into one text blob for vectorization."""
    subject = subject or ""
    body = body or ""
    return f"{subject} {subject} {body}".strip()  # subject weighted x2
