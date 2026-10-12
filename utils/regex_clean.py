# utils/regex_clean.py
import re
from datetime import datetime
from typing import List, Dict

SECTION_RE = re.compile(r"Section\s+(\d+[A-Za-z-]*)\s+of\s+([A-Za-z .()/-]+)", re.I)
DATE_PATTERNS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y"]

COURT_ALIASES = {
    r"\bsc\b|supreme\s*court\b": "Supreme Court of India",
    r"delhi\s*high\s*court|d\s*hc\b": "Delhi High Court",
    r"bombay\s*high\s*court": "Bombay High Court",
    r"madras\s*high\s*court": "Madras High Court",
    r"calcutta\s*high\s*court": "Calcutta High Court",
}

ACT_NORMALIZE = {
    r"\bipc\b|indian\s*penal\s*code": "IPC",
    r"\bcrpc\b|code\s*of\s*criminal\s*procedure": "CrPC",
    r"ndps\b|narcotic\b[\w\s]*psychotropic": "NDPS Act",
    r"pocso\b|protection\s*of\s*children\s*from\s*sexual\s*offences": "POCSO Act",
}


def to_iso_date(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    for fmt in DATE_PATTERNS:
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", t)
    if m:
        try:
            return datetime.strptime(" ".join(m.groups()), "%d %B %Y").strftime("%Y-%m-%d")
        except Exception:
            try:
                return datetime.strptime(" ".join(m.groups()), "%d %b %Y").strftime("%Y-%m-%d")
            except Exception:
                pass
    m2 = re.search(r"(\d{4})[^\d]?(\d{1,2})[^\d]?(\d{1,2})", t)
    if m2:
        y, mo, d = m2.groups()
        try:
            return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
        except Exception:
            pass
    return t


def normalize_court(court: str) -> str:
    if not court:
        return ""
    s = court.lower()
    for pat, name in COURT_ALIASES.items():
        if re.search(pat, s):
            return name
    return court.strip()


def extract_sections(text: str, acts_list: List[str]) -> List[Dict[str, str]]:
    sections = []
    if text:
        for sec, act in SECTION_RE.findall(text):
            act_clean = act.strip()
            for pat, canon in ACT_NORMALIZE.items():
                if re.search(pat, act_clean, re.I):
                    act_clean = canon
                    break
            sections.append({"section": sec.strip(), "act": act_clean})
    for a in acts_list or []:
        m = SECTION_RE.search(a)
        if m:
            sec, act = m.groups()
            act_clean = act.strip()
            for pat, canon in ACT_NORMALIZE.items():
                if re.search(pat, act_clean, re.I):
                    act_clean = canon
                    break
            e = {"section": sec.strip(), "act": act_clean}
            if e not in sections:
                sections.append(e)
    return sections


def standardize_decision(label: str, conclusion: str) -> str:
    s = (label or "").upper().strip()
    if s in {"GRANTED", "REJECTED"}:
        return s
    text = (conclusion or "")
    if re.search(r"\b(bail\s+is\s+granted|grant\s+of\s+bail)\b", text, re.I):
        return "GRANTED"
    if re.search(r"\b(bail\s+is\s+rejected|reject\s+the\s+bail)\b", text, re.I):
        return "REJECTED"
    return "UNKNOWN"
