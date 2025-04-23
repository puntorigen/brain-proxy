from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional
import re
from dateparser.search import search_dates

# Map of multilingual relative words → span type
_REL_MAP = {
    # single‑day offsets
    "yesterday": -1, "ayer": -1, "hier": -1, "ontem": -1, "昨日": -1,
    "today": 0, "hoy": 0, "aujourd'hui": 0, "hoje": 0, "今天": 0,
    # previous week
    "last week": -7, "semana pasada": -7, "la semaine dernière": -7,
    "vorige week": -7, "上周": -7,
    # previous month (approx)
    "last month": -30, "mes pasado": -30, "monate zuvor": -30, "上个月": -30,
}


def _span_from_relative(token: str):
    token = token.lower()
    delta = _REL_MAP[token]
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    if delta == 0:  # today
        return today, today + timedelta(days=1, seconds=-1)
    if delta == -1:  # yesterday
        start = today - timedelta(days=1)
        return start, start + timedelta(days=1, seconds=-1)
    if delta == -7:  # last week (Mon‑Sun)
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return start, end
    if delta == -30:  # last month (approx)
        first_this_month = today.replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)
        start = last_prev_month.replace(day=1)
        end = last_prev_month.replace(hour=23, minute=59, second=59)
        return start, end
    return None


def extract_timerange(text: str) -> Optional[Tuple[datetime, datetime]]:
    """Return (start,end) UTC if *text* contains a date or relative phrase."""
    lower = text.lower()
    for token in _REL_MAP:
        if token in lower:
            return _span_from_relative(token)

    # search_dates handles explicit dates in >200 languages
    found = search_dates(text, settings={"RELATIVE_BASE": datetime.now(timezone.utc)})
    if found:
        _, dt = found[0]
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end = start + timedelta(days=1, seconds=-1)
        return start, end

    return None
