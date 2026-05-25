"""
market_calendar.py — NSE market calendar: dynamic holiday fetch + open/closed logic.

Rules (strict):
  - NO hardcoded holidays
  - Fetches from NSE website dynamically
  - Caches result to avoid repeated scraping (file cache + in-process cache)
  - is_market_open() is the single gatekeeper used by predictor and UI
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.utils import DATA_DIR, get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cache file lives in data/
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_FILE = DATA_DIR / "nse_holidays_cache.json"
_MAX_CACHE_AGE_DAYS = 30   # re-fetch at most once a month

# In-process cache so repeated calls within one Streamlit session are free
_MEM_CACHE: dict[int, dict[str, str]] = {}

# NSE holiday calendar URL (CM segment — equity)
_NSE_HOLIDAY_URL = (
    "https://www.nseindia.com/api/holiday-master?type=trading"
)
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_nse_holidays(year: int) -> dict[str, str]:
    """
    Return a dict of {ISO-date-string: holiday-name} for *year*.

    e.g. {"2026-04-14": "Dr. Ambedkar Jayanti", ...}

    Fetch strategy (in order):
      1. In-process memory cache
      2. File cache (data/nse_holidays_cache.json, max 30 days old)
      3. NSE API (requests + json parse)
      4. Fallback: NSE website HTML scrape
    If all fail, returns {} with a warning — the system never crashes.
    """
    if year in _MEM_CACHE:
        return _MEM_CACHE[year]

    # Try file cache first
    cached = _load_file_cache(year)
    if cached is not None:
        _MEM_CACHE[year] = cached
        return cached

    # Try live fetch
    holidays = _fetch_from_nse_api(year)
    if not holidays:
        holidays = _fetch_from_nse_html(year)

    if holidays:
        _save_file_cache(year, holidays)
        _MEM_CACHE[year] = holidays
        log.info("NSE holidays for %d: %d holidays fetched.", year, len(holidays))
    else:
        log.warning("Could not fetch NSE holidays for %d — treating all weekdays as open.", year)
        holidays = {}
        _MEM_CACHE[year] = holidays

    return holidays


def is_market_open(target_date: date) -> tuple[bool, Optional[str]]:
    """
    Determine if NSE is open on *target_date*.

    Returns
    -------
    (True,  None)            — market is open
    (False, "Weekend")       — Saturday or Sunday
    (False, "<Holiday Name>")— NSE trading holiday
    """
    # Weekend check
    if target_date.weekday() >= 5:   # 5=Saturday, 6=Sunday
        return False, "Weekend"

    # Holiday check
    holidays = get_nse_holidays(target_date.year)
    iso = target_date.isoformat()   # "2026-04-14"
    if iso in holidays:
        return False, holidays[iso]

    return True, None


def get_holidays_for_month(year: int, month: int) -> list[dict]:
    """
    Return a list of {date, name} dicts for holidays in the given month.
    Useful for the UI holiday calendar page.
    """
    all_holidays = get_nse_holidays(year)
    result = []
    for iso_date, name in sorted(all_holidays.items()):
        try:
            d = date.fromisoformat(iso_date)
            if d.year == year and d.month == month:
                result.append({"date": d, "name": name})
        except ValueError:
            continue
    return result


def get_all_holidays_for_year(year: int) -> list[dict]:
    """Return all holidays for the year as sorted list of {date, name}."""
    all_holidays = get_nse_holidays(year)
    result = []
    for iso_date, name in sorted(all_holidays.items()):
        try:
            d = date.fromisoformat(iso_date)
            result.append({
                "date"      : d,
                "name"      : name,
                "day"       : d.strftime("%A"),
            })
        except ValueError:
            continue
    return result


# ─────────────────────────────────────────────────────────────────────────────
# NSE API fetch  (JSON endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_nse_api(year: int) -> dict[str, str]:
    """
    Hit the NSE holiday-master JSON API.
    Returns {iso_date: holiday_name} or {} on failure.
    """
    try:
        import requests

        session = requests.Session()
        # NSE requires a cookie from the main page first
        session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)

        resp = session.get(_NSE_HOLIDAY_URL, headers=_NSE_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # The response has a key "CM" (Capital Market) with a list of dicts
        cm_holidays = data.get("CM", [])
        if not cm_holidays:
            # Try top-level list
            cm_holidays = data if isinstance(data, list) else []

        holidays: dict[str, str] = {}
        for item in cm_holidays:
            # Fields vary: tradingDate / date / holidayDate
            raw_date = (
                item.get("tradingDate")
                or item.get("date")
                or item.get("holidayDate")
                or ""
            )
            name = (
                item.get("description")
                or item.get("holidayName")
                or item.get("name")
                or "Holiday"
            )
            iso = _parse_nse_date(raw_date)
            if iso:
                parsed_year = int(iso[:4])
                if parsed_year == year:
                    holidays[iso] = name.strip()

        return holidays

    except Exception as exc:
        log.warning("NSE API fetch failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# NSE HTML scrape fallback
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_nse_html(year: int) -> dict[str, str]:
    """
    Scrape the NSE holiday page HTML as a fallback.
    Returns {iso_date: holiday_name} or {} on failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://www.nseindia.com/market-data/holiday-calendar"
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)
        resp = session.get(url, headers=_NSE_HEADERS, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        holidays: dict[str, str] = {}

        # Look for table rows with date patterns
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                raw_date = cells[0].get_text(strip=True)
                name     = cells[1].get_text(strip=True)
                iso = _parse_nse_date(raw_date)
                if iso:
                    parsed_year = int(iso[:4])
                    if parsed_year == year:
                        holidays[iso] = name

        return holidays

    except Exception as exc:
        log.warning("NSE HTML scrape failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Date parser — handles multiple NSE date formats
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%d-%b-%Y",   # 14-Apr-2026
    "%d-%b-%y",   # 14-Apr-26
    "%d/%m/%Y",   # 14/04/2026
    "%Y-%m-%d",   # 2026-04-14
    "%d %b %Y",   # 14 Apr 2026
    "%B %d, %Y",  # April 14, 2026
    "%d-%m-%Y",   # 14-04-2026
]


def _parse_nse_date(raw: str) -> Optional[str]:
    """Convert a raw date string to ISO format YYYY-MM-DD, or None on failure."""
    raw = raw.strip()
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# File cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_file_cache(year: int) -> Optional[dict[str, str]]:
    if not _CACHE_FILE.exists():
        return None
    try:
        age_days = (date.today() - date.fromtimestamp(_CACHE_FILE.stat().st_mtime)).days
        if age_days > _MAX_CACHE_AGE_DAYS:
            return None
        with _CACHE_FILE.open() as f:
            all_data = json.load(f)
        return all_data.get(str(year))
    except Exception:
        return None


def _save_file_cache(year: int, holidays: dict[str, str]) -> None:
    try:
        all_data: dict = {}
        if _CACHE_FILE.exists():
            with _CACHE_FILE.open() as f:
                all_data = json.load(f)
        all_data[str(year)] = holidays
        with _CACHE_FILE.open("w") as f:
            json.dump(all_data, f, indent=2)
    except Exception as exc:
        log.warning("Could not write holiday cache: %s", exc)


def invalidate_cache() -> None:
    """Force a fresh fetch next time (used in testing / manual override)."""
    _MEM_CACHE.clear()
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
    log.info("Holiday cache invalidated.")
