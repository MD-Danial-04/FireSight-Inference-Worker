import re

NATO_LETTERS: dict[str, str] = {
    "alfa": "A",
    "alpha": "A",
    "bravo": "B",
    "charlie": "C",
    "delta": "D",
    "echo": "E",
    "foxtrot": "F",
    "golf": "G",
    "hotel": "H",
    "india": "I",
    "juliet": "J",
    "juliett": "J",
    "kilo": "K",
    "lima": "L",
    "mike": "M",
    "november": "N",
    "oscar": "O",
    "papa": "P",
    "quebec": "Q",
    "romeo": "R",
    "sierra": "S",
    "tango": "T",
    "uniform": "U",
    "victor": "V",
    "whiskey": "W",
    "whisky": "W",
    "xray": "X",
    "yankee": "Y",
    "zulu": "Z",
}

NATO_DIGITS: dict[str, str] = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "wun": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "fower": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "niner": "9",
}

RANK_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btriple\s+s\b", re.IGNORECASE), "SSS"),
    (re.compile(r"\bsergeant\s+3\b", re.IGNORECASE), "SGT3"),
    (re.compile(r"\bsgt\.?\s*3\b", re.IGNORECASE), "SGT3"),
    (re.compile(r"\bsergeant\s+2\b", re.IGNORECASE), "SGT2"),
    (re.compile(r"\bsgt\.?\s*2\b", re.IGNORECASE), "SGT2"),
    (re.compile(r"\bsergeant\s+1\b", re.IGNORECASE), "SGT1"),
    (re.compile(r"\bsgt\.?\s*1\b", re.IGNORECASE), "SGT1"),
]

_CALL_SIGN_PATTERN = re.compile(
    r"\bLF[-\s]?(\d{3})\b|\bLF[-\s]?A(\d{2})\b",
    re.IGNORECASE,
)
_NPC_PATTERN = re.compile(r"\b([\w-]+\s+NPC)\b", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_LIAISE_MISHEAR_PATTERN = re.compile(
    r"\b(liars?|lease|lies|lyase|liaise?)\s+(with\b)",
    re.IGNORECASE,
)
_LIAISE_CLAUSE_PATTERN = re.compile(
    r"\bLiase with\b(.+?)(?=\.\s+Case\b|\.\s*$)",
    re.IGNORECASE | re.DOTALL,
)


def _hint_call_sign(text: str) -> str:
    match = _CALL_SIGN_PATTERN.search(text)
    if not match:
        return ""
    if match.group(1):
        return f"LF{match.group(1)}"
    return f"LF8{match.group(2)}"


def _normalize_call_sign(current: str, source_text: str) -> str:
    for text in (current, source_text):
        hinted = _hint_call_sign(text)
        if hinted:
            return hinted
    return current.strip()


def _normalize_location(location: str) -> str:
    if not location.strip():
        return location
    result = re.sub(r"\bGall\b", "Gul", location, flags=re.IGNORECASE)
    result = re.sub(r"\bAvenue\b", "Ave", result, flags=re.IGNORECASE)
    return result.strip()


def _normalize_liaise_text(text: str) -> str:
    return _LIAISE_MISHEAR_PATTERN.sub(r"Liase \2", text)


def _extract_liaise_clause(text: str) -> str:
    normalized = _normalize_liaise_text(text)
    match = _LIAISE_CLAUSE_PATTERN.search(normalized)
    if not match:
        return ""
    detail = match.group(1).strip().rstrip(",")
    return f"Liase with {detail}"


def _normalize_events_circumstances(current: str, source_text: str) -> str:
    current = _normalize_liaise_text(current)
    liaise = _extract_liaise_clause(source_text)
    if not liaise:
        return current.strip()

    if re.search(r"\bliase with\b", current, re.IGNORECASE):
        return current.strip()

    if current.strip():
        return f"{current.strip().rstrip('.')}. {liaise}"
    return liaise


def _normalize_ranks(text: str) -> str:
    result = text
    for pattern, replacement in RANK_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def _token_to_nato_char(token: str) -> str | None:
    lower = token.lower()
    if lower in NATO_LETTERS:
        return NATO_LETTERS[lower]
    if lower in NATO_DIGITS:
        return NATO_DIGITS[lower]
    if len(token) == 1 and token.isalpha():
        return token.upper()
    if token.isdigit():
        return token
    return None


def _decode_nato_service_id(text: str) -> str:
    tokens = _TOKEN_PATTERN.findall(text)
    best = ""
    current = ""

    for token in tokens:
        if token.lower() == "or":
            continue
        char = _token_to_nato_char(token)
        if char is not None:
            current += char
            continue
        if len(current) > len(best):
            best = current
        current = ""

    if len(current) > len(best):
        best = current

    return best


def _contains_spoken_rank_or_nato(text: str) -> bool:
    lower = text.lower()
    if any(pattern.search(text) for pattern, _ in RANK_REPLACEMENTS):
        return True
    return any(word in lower for word in (*NATO_LETTERS.keys(), *NATO_DIGITS.keys()))


def _strip_nato_spelling(text: str) -> str:
    tokens = _TOKEN_PATTERN.findall(text)
    if not tokens:
        return text

    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0

    for index, token in enumerate(tokens):
        if _token_to_nato_char(token) is not None:
            if cur_len == 0:
                cur_start = index
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_len = 0
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len

    nato_indices = (
        set(range(best_start, best_start + best_len)) if best_len else set()
    )
    kept = [token for index, token in enumerate(tokens) if index not in nato_indices]
    result = " ".join(kept)
    return re.sub(r"\s+", " ", result).strip().strip(".,;")


def _normalize_handover_officer(officer: str, source_text: str) -> str:
    text = officer.strip() or source_text
    if not text:
        return officer

    if not _contains_spoken_rank_or_nato(text):
        return officer.strip()

    normalized = _normalize_ranks(text)
    service_id = _decode_nato_service_id(normalized)

    if service_id:
        without_nato = _strip_nato_spelling(normalized)
        if service_id in without_nato:
            normalized = without_nato
        else:
            normalized = f"{without_nato} {service_id}".strip()
    else:
        normalized = re.sub(r"\s+", " ", normalized).strip()

    return re.sub(r"\s+", " ", normalized).strip()


def _hint_handover_npc(text: str) -> str:
    match = _NPC_PATTERN.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def apply_field_normalization(fields: dict[str, str], source_text: str) -> dict[str, str]:
    source_text = _normalize_liaise_text(source_text)
    result = dict(fields)

    result["applianceCallSign"] = _normalize_call_sign(
        result.get("applianceCallSign", ""),
        source_text,
    )
    result["locationOfFire"] = _normalize_location(result.get("locationOfFire", ""))
    result["eventsCircumstances"] = _normalize_events_circumstances(
        result.get("eventsCircumstances", ""),
        source_text,
    )

    result["handoverOfficer"] = _normalize_handover_officer(
        result.get("handoverOfficer", ""),
        source_text,
    )

    if not result.get("handoverNpc"):
        hinted = _hint_handover_npc(source_text)
        if hinted:
            result["handoverNpc"] = hinted

    return result
