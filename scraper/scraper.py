import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from readability import Document
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


ROOT = Path(__file__).resolve().parents[1]
URLS_FILE = ROOT / "scraper" / "urls.json"
RAW_DIR = ROOT / "data" / "raw"
RAW_OUTPUT = RAW_DIR / "raw_documents.json"
FAILED_OUTPUT = RAW_DIR / "failed_urls.json"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "scraper.log"
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "250"))
INSECURE_SSL_DOMAINS = {
    "bbmp.gov.in",
}
ALTERNATE_URLS = {
    "https://sakala.karnataka.gov.in/": [
        "http://sakala.kar.nic.in",
    ],
    "https://bbmp.gov.in/": [
        "https://www.bbmp.gov.in/",
        "http://bbmp.gov.in/",
    ],
}

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scraper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"HEAD", "GET", "OPTIONS"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def load_urls(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"urls file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    entries: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                entries.append({"url": item})
            elif isinstance(item, dict) and item.get("url"):
                entries.append(item)
    elif isinstance(payload, dict) and isinstance(payload.get("urls"), list):
        for item in payload["urls"]:
            if isinstance(item, str):
                entries.append({"url": item})
            elif isinstance(item, dict) and item.get("url"):
                entries.append(item)

    if not entries:
        raise ValueError("No valid URLs found in scraper/urls.json")
    return entries


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_text_from_html(html: str) -> Dict[str, str]:
    doc = Document(html)
    title = doc.short_title() or ""
    main_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(main_html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    cleaned = normalize_text(text)
    return {"title": normalize_text(title), "text": cleaned}


def _fetch_once(
    session: requests.Session, url: str, timeout: int = 35, verify: bool = True
) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        )
    }
    response = session.get(url, headers=headers, timeout=timeout, verify=verify)
    response.raise_for_status()
    return response.text


def fetch_url_with_fallbacks(
    session: requests.Session, url: str, logger: logging.Logger, timeout: int = 35
) -> Dict[str, Any]:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    candidates = [url] + ALTERNATE_URLS.get(url, [])
    attempts: List[Dict[str, str]] = []

    for candidate in candidates:
        try:
            html = _fetch_once(session, candidate, timeout=timeout, verify=True)
            return {"html": html, "resolved_url": candidate, "attempts": attempts}
        except requests.exceptions.SSLError as exc:
            attempts.append({"url": candidate, "error": f"ssl_error: {exc}"})
            if host in INSECURE_SSL_DOMAINS:
                logger.warning("Retrying with verify=False for %s", candidate)
                try:
                    html = _fetch_once(session, candidate, timeout=timeout, verify=False)
                    return {"html": html, "resolved_url": candidate, "attempts": attempts}
                except Exception as ssl_retry_exc:  # noqa: BLE001
                    attempts.append({"url": candidate, "error": f"ssl_insecure_failed: {ssl_retry_exc}"})
        except Exception as exc:  # noqa: BLE001
            attempts.append({"url": candidate, "error": str(exc)})

    raise RuntimeError(f"All attempts failed for {url}: {attempts[-1]['error'] if attempts else 'unknown error'}")


def scrape_documents(
    url_entries: List[Dict[str, Any]], logger: logging.Logger
) -> Dict[str, List[Dict[str, Any]]]:
    session = build_session()
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, entry in enumerate(tqdm(url_entries, desc="Scraping URLs"), start=1):
        url = entry["url"].strip()
        category = entry.get("category", "")
        doc_type = entry.get("type", "")

        try:
            fetched = fetch_url_with_fallbacks(session, url, logger)
            html = fetched["html"]
            resolved_url = fetched["resolved_url"]
            if not html:
                logger.warning("Empty response for %s", url)
                failures.append({"url": url, "reason": "empty_response"})
                continue
            parsed = extract_text_from_html(html)
            if not parsed["text"]:
                logger.warning("No text extracted for %s", url)
                failures.append({"url": url, "reason": "empty_extraction", "resolved_url": resolved_url})
                continue
            if len(parsed["text"]) < MIN_TEXT_LENGTH:
                logger.warning("Low text extraction for %s (len=%s)", url, len(parsed["text"]))
                failures.append(
                    {
                        "url": url,
                        "reason": "low_content",
                        "text_length": len(parsed["text"]),
                        "resolved_url": resolved_url,
                    }
                )
                continue

            parsed_url = urlparse(resolved_url)
            record = {
                "doc_id": f"doc_{i:05d}",
                "url": url,
                "resolved_url": resolved_url,
                "domain": parsed_url.netloc,
                "title": parsed["title"] or entry.get("title", ""),
                "category": category,
                "doc_type": doc_type,
                "text": parsed["text"],
                "text_length": len(parsed["text"]),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
            records.append(record)
            logger.info("Scraped %s (len=%s)", url, record["text_length"])
        except Exception as exc:
            logger.exception("Failed scraping %s: %s", url, exc)
            failures.append({"url": url, "reason": "request_failed", "error": str(exc)})

    return {"records": records, "failures": failures}


def write_output(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def write_failures(failures: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)


def main() -> None:
    logger = setup_logger()
    logger.info("Starting scraper...")
    urls = load_urls(URLS_FILE)
    logger.info("Loaded %s URL entries", len(urls))

    result = scrape_documents(urls, logger)
    records = result["records"]
    failures = result["failures"]
    write_output(records, RAW_OUTPUT)
    write_failures(failures, FAILED_OUTPUT)
    logger.info("Saved %s documents to %s", len(records), RAW_OUTPUT)
    logger.info("Saved %s failed URLs to %s", len(failures), FAILED_OUTPUT)


if __name__ == "__main__":
    main()
