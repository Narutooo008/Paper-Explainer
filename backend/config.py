"""Application settings and environment configuration."""
import os


def quality_mode() -> str:
    """Return 'fast' or 'normal' (default) based on the QUALITY environment variable."""
    val = os.getenv("QUALITY", "normal").strip().lower()
    return "fast" if val == "fast" else "normal"


def site_url() -> str:
    """Return normalized base site URL without trailing slash (default: http://localhost:8000)."""
    return os.getenv("SITE_URL", "http://localhost:8000").strip().rstrip("/")


def site_name() -> str:
    """Return site brand name (default: PaperExplainer)."""
    return os.getenv("SITE_NAME", "PaperExplainer").strip()


def contact_email() -> str:
    """Return public contact email."""
    return os.getenv("CONTACT_EMAIL", "contact@paperexplainer.org").strip()


def build_url(path: str = "") -> str:
    """Build absolute URL from SITE_URL, avoiding hardcoded domains."""
    base = site_url()
    if not path or path == "/":
        return f"{base}/"
    clean_path = "/" + path.lstrip("/")
    return f"{base}{clean_path}"
