"""Stateless text cleaning functions shared across all text-based detectors."""
import re
import html


def clean_text(text: str) -> str:
    """Normalize a raw text message or email body for ML consumption."""
    text = html.unescape(text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize URLs to a token so the model sees the pattern, not the domain
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    # Normalize phone numbers
    text = re.sub(r"\+?[\d][\d\s\-().]{7,}\d", " PHONE ", text)
    # Normalize dollar amounts
    text = re.sub(r"\$[\d,]+(\.\d+)?", " MONEY ", text)
    # Lowercase and collapse whitespace
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_email_parts(raw_email: str) -> dict:
    """Split a raw email string into header fields and body.

    Returns a dict with keys: subject, sender, body, cleaned_body.
    If the input is a plain text message (no headers), body == raw_email.
    """
    subject = ""
    sender = ""
    body = raw_email

    # Detect if there are RFC-style headers (heuristic: first line contains ":")
    lines = raw_email.split("\n")
    header_end = 0
    in_headers = True
    for i, line in enumerate(lines):
        if in_headers:
            if line.strip() == "":
                header_end = i
                body = "\n".join(lines[i + 1:])
                in_headers = False
            elif line.lower().startswith("subject:"):
                subject = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("from:"):
                sender = line.split(":", 1)[-1].strip()

    return {
        "subject": subject,
        "sender": sender,
        "body": body,
        "cleaned_body": clean_text(subject + " " + body),
    }
