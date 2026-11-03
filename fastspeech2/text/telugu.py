"""Rule-based Telugu text front-end (new file, not in the reference implementation).

Telugu script is close to phonemic, so a grapheme-to-phoneme converter can be written
as a small set of rules over the Unicode akshara structure:

  consonant (+ virama)        -> consonant only          క్  -> k
  consonant (+ vowel sign)    -> consonant + that vowel   కా  -> k aa
  bare consonant              -> consonant + inherent /a/ క   -> k a
  anusvara (ం)                -> nasal homorganic with the next consonant, /m/ otherwise

Numbers, times and ordinals (e.g. "12వ") are expanded to Telugu words first.
The phone labels are lower-case ASCII so they never collide with upper-case ARPAbet.
"""

import re
import unicodedata

VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
CONSONANTS = [
    "k", "kh", "g", "gh", "ng",
    "c", "ch", "j", "jh", "nj",
    "tx", "txh", "dx", "dxh", "nx",
    "t", "th", "d", "dh", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "l", "lx", "v",
    "sh", "sx", "s", "h",
]
PHONES = VOWELS + CONSONANTS

_INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"],
    "ఉ": ["u"], "ఊ": ["uu"], "ఋ": ["r", "u"], "ౠ": ["r", "uu"],
    "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"],
    "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
_VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"],
    "ూ": ["uu"], "ృ": ["r", "u"], "ౄ": ["r", "uu"],
    "ౢ": ["l", "u"], "ౣ": ["l", "uu"],
    "ె": ["e"], "ే": ["ee"], "ై": ["ai"],
    "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}
_CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "nj",
    "ట": "tx", "ఠ": "txh", "డ": "dx", "ఢ": "dxh", "ణ": "nx",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "lx",
    "ఴ": "lx", "వ": "v",
    "శ": "sh", "ష": "sx", "స": "s", "హ": "h",
    "ౘ": "c", "ౙ": "j", "ౚ": "r",
}
_VIRAMA = "్"
_ANUSVARA = "ం"
_VISARGA = "ః"
_TELUGU_DIGITS = {chr(0x0C66 + i): str(i) for i in range(10)}

# place of articulation of the consonant that follows an anusvara
_NASAL_FOR = {}
for _p in ["k", "kh", "g", "gh"]:
    _NASAL_FOR[_p] = "ng"
for _p in ["c", "ch", "j", "jh"]:
    _NASAL_FOR[_p] = "nj"
for _p in ["tx", "txh", "dx", "dxh"]:
    _NASAL_FOR[_p] = "nx"
for _p in ["t", "th", "d", "dh", "n"]:
    _NASAL_FOR[_p] = "n"
_ANUSVARA_MARK = "<ANUSVARA>"


def word_to_phones(word):
    """Convert one Telugu word (no spaces) to a list of phones."""
    word = unicodedata.normalize("NFC", word)
    phones = []
    pending = None  # consonant still waiting to learn which vowel follows it

    def flush(vowel=("a",)):
        nonlocal pending
        if pending is not None:
            phones.append(pending)
            phones.extend(vowel)
            pending = None

    for ch in word:
        if ch in _CONSONANTS:
            flush()
            pending = _CONSONANTS[ch]
        elif ch in _VOWEL_SIGNS:
            if pending is not None:
                flush(_VOWEL_SIGNS[ch])
            else:
                phones.extend(_VOWEL_SIGNS[ch])
        elif ch == _VIRAMA:
            flush(())
        elif ch in _INDEPENDENT_VOWELS:
            flush()
            phones.extend(_INDEPENDENT_VOWELS[ch])
        elif ch == _ANUSVARA:
            flush()
            phones.append(_ANUSVARA_MARK)
        elif ch == _VISARGA:
            flush()
            phones.append("h")
        # candrabindu, avagraha, ZWJ/ZWNJ, nukta and anything else are silent
    flush()

    for i, p in enumerate(phones):
        if p == _ANUSVARA_MARK:
            nxt = phones[i + 1] if i + 1 < len(phones) else None
            phones[i] = _NASAL_FOR.get(nxt, "m")
    return phones


# ---------------------------------------------------------------- numbers

_SMALL = [
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
    "పద్దెనిమిది", "పందొమ్మిది",
]
_TENS = {
    2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై",
    6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై",
}


def _below_100(n):
    if n < 20:
        return _SMALL[n]
    t, u = divmod(n, 10)
    return _TENS[t] if u == 0 else _TENS[t] + " " + _SMALL[u]


def _below_1000(n):
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    if h == 1:
        head = "వంద" if r == 0 else "నూట"
    else:
        head = _below_100(h) + (" వందలు" if r == 0 else " వందల")
    return head if r == 0 else head + " " + _below_100(r)


def cardinal(n):
    """Telugu cardinal number words, e.g. 2026 -> రెండు వేల ఇరవై ఆరు."""
    if n < 1000:
        return _below_1000(n)
    if n < 100000:
        th, r = divmod(n, 1000)
        head = "వెయ్యి" if th == 1 else _below_100(th) + (" వేలు" if r == 0 else " వేల")
        return head if r == 0 else head + " " + _below_1000(r)
    if n < 10 ** 7:
        lk, r = divmod(n, 100000)
        head = "లక్ష" if lk == 1 else _below_100(lk) + (" లక్షలు" if r == 0 else " లక్షల")
        return head if r == 0 else head + " " + cardinal(r)
    cr, r = divmod(n, 10 ** 7)
    head = "కోటి" if cr == 1 else cardinal(cr) + (" కోట్లు" if r == 0 else " కోట్ల")
    return head if r == 0 else head + " " + cardinal(r)


def ordinal(n):
    """Ordinal used before nouns, e.g. 12 -> పన్నెండవ (as in "12వ తేదీ")."""
    words = cardinal(n)
    if words.endswith("ై"):  # ై : ఇరవై -> ఇరవయ్యవ
        return words[:-1] + "య్యవ"
    if words[-1] in ("ు", "ి"):  # ు / ి : పన్నెండు -> పన్నెండవ
        return words[:-1] + "వ"
    return words + "వ"


_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
_ORDINAL_RE = re.compile(r"(\d+)వ(?![ా-్])")  # digits + bare వ
_NUMBER_RE = re.compile(r"\d+")


def _time(m):
    h, mins = int(m.group(1)), int(m.group(2))
    return cardinal(h) if mins == 0 else cardinal(h) + " " + cardinal(mins)


def normalize(text):
    """NFC-normalise, map Telugu digits to ASCII and spell out numbers in Telugu."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(_TELUGU_DIGITS.get(c, c) for c in text)
    text = text.replace("‌", "").replace("‍", "").replace("﻿", "")
    # IndicTTS transcripts sometimes type a Latin "o" for the look-alike anusvara (వీళుoటే)
    text = re.sub(r"(?<=[ఀ-౿])o", _ANUSVARA, text)
    text = _TIME_RE.sub(_time, text)
    text = _ORDINAL_RE.sub(lambda m: ordinal(int(m.group(1))), text)
    text = _NUMBER_RE.sub(lambda m: cardinal(int(m.group(0))), text)
    return re.sub(r"\s+", " ", text).strip()


_WORD_RE = re.compile(r"[ఀ-౿‌‍]+")


def words(text):
    """Telugu words of an already-normalised sentence (punctuation and Latin dropped)."""
    return _WORD_RE.findall(text)


def text_to_words(text):
    """Sentence -> [(romanised word, phones)], with ("sp", ["sp"]) at punctuation.

    Words are labelled in romanised form because matplotlib cannot shape Telugu script.
    """
    text = normalize(text)
    out = []
    for token in re.findall(r"[ఀ-౿‌‍]+|[,.;:!?।॥]", text):
        if _WORD_RE.fullmatch(token):
            phones = word_to_phones(token)
            if phones:
                out.append(("".join(phones), phones))
        elif out and out[-1][0] != "sp":
            out.append(("sp", ["sp"]))
    while out and out[-1][0] == "sp":
        out.pop()
    return out


def text_to_phones(text):
    """Sentence -> phone list, with 'sp' at punctuation marks (used at inference time)."""
    return [p for _, phones in text_to_words(text) for p in phones]
