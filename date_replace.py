

import re
import datetime

# Map month names
MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}

SP = r"[ \u202f]"
DAY_ORD = r"\d{1,2}(?:st|nd|rd|th)?"

DATE_PATTERN = re.compile(
    rf"""
    (
        # January 3, 2021  /  Jan 3rd, 2021  (full date)
        (?P<month_name>(?:Jan|January|Feb|February|Mar|March|Apr|April|
                        May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|
                        Oct|October|Nov|November|Dec|December))
        {SP}+
        (?P<day_name>{DAY_ORD})
        (?:,)?{SP}*
        (?P<year_name>\d{{2,4}})
    )
    |
    (
        # 3 Jan 2021  /  3rd Jan 2021 (full date)
        (?P<day2>{DAY_ORD})
        {SP}+
        (?P<month2>(?:Jan|January|Feb|February|Mar|March|Apr|April|
                    May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|
                    Oct|October|Nov|November|Dec|December))
        {SP}+
        (?P<year2>\d{{2,4}})
    )
    |
    (
        # NEW: January 3  (no year)
        (?P<month_name_only>(?:Jan|January|Feb|February|Mar|March|Apr|April|
                             May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|
                             Oct|October|Nov|November|Dec|December))
        {SP}+
        (?P<day_only1>{DAY_ORD})
    )
    |
    (
        # NEW: 3 January  (no year)
        (?P<day_only2>{DAY_ORD})
        {SP}+
        (?P<month_only>(?:Jan|January|Feb|February|Mar|March|Apr|April|
                        May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|
                        Oct|October|Nov|November|Dec|December))
    )
    |
    (
        # 2021-01-03 or 2021/1/3
        (?P<y_iso>\d{{4}})
        [-/]
        (?P<m_iso>\d{{1,2}})
        [-/]
        (?P<d_iso>\d{{1,2}})
    )
    |
    (
        # 1/3/21 or 01/03/2021
        (?P<m_slash>\d{{1,2}})
        /
        (?P<d_slash>\d{{1,2}})
        /
        (?P<y_slash>\d{{2,4}})
    )
    """,
    re.IGNORECASE | re.VERBOSE
)


def strip_ordinal(s: str) -> str:
    return re.sub(r"(st|nd|rd|th)$", "", s, flags=re.IGNORECASE)

def reformat(date_obj: datetime.date, match: re.Match) -> str:
    g = match.groupdict()

    def sp():  # detect whether match used space or \u202f
        return "\u202f" if "\u202f" in match.group(0) else " "

    # --- Month-name formats: January 3, 2021 ---
    if g["month_name"]:
        month_raw = g["month_name"]
        # abbreviated vs full
        month_str = date_obj.strftime("%b" if len(month_raw) <= 3 else "%B")

        # preserve comma presence
        comma = "," if "," in match.group(0) else ""

        # always plain numeric day (no ordinal)
        return f"{month_str}{sp()}{date_obj.day}{comma}{sp()}{date_obj.year}"

    # --- Day Month Year: 3 Jan 2021 OR 3rd Jan 2021 ---
    if g["day2"]:
        month_raw = g["month2"]
        month_str = date_obj.strftime("%b" if len(month_raw) <= 3 else "%B")
        return f"{date_obj.day}{sp()}{month_str}{sp()}{date_obj.year}"

    # --- ISO 2021-01-03 ---
    if g["y_iso"]:
        sep = "-" if "-" in match.group(0) else "/"
        return f"{date_obj.year}{sep}{date_obj.month:02d}{sep}{date_obj.day:02d}"

    # --- Slash 1/3/21 or 01/03/2021 ---
    if g["m_slash"]:
        year_fmt = "%y" if len(g["y_slash"]) == 2 else "%Y"
        year_str = date_obj.strftime(year_fmt)
        return f"{date_obj.month}/{date_obj.day}/{year_str}"

    # Month Day (no year)
    if g["month_name_only"]:
        month_raw = g["month_name_only"]
        month_str = date_obj.strftime("%b" if len(month_raw) <= 3 else "%B")
        return f"{month_str}{sp()}{date_obj.day}"

    # Day Month (no year)
    if g["day_only2"]:
        month_raw = g["month_only"]
        month_str = date_obj.strftime("%b" if len(month_raw) <= 3 else "%B")
        return f"{date_obj.day}{sp()}{month_str}"

    return match.group(0)

def detect_space(match: re.Match) -> str:
    """
    Detect whether the match used a normal space or U+202F thin NBSP.
    Default to space if unclear.
    """
    m = match.group(0)
    return "\u202f" if "\u202f" in m else " "


def replace_dates(text: str, date_obj: datetime.date) -> str:
    """
    Replace all dates in `text` with the same-format representation of `date_obj`.
    """
    return DATE_PATTERN.sub(lambda m: reformat(date_obj, m), text)

def contains_date(text: str) -> bool:
    return DATE_PATTERN.search(text) is not None