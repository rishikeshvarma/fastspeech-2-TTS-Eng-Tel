"""English text front-end used at inference time (new file).

The reference synthesize.py looked every word up in a lexicon first and only used g2p_en
for unknown words. That gives "conduct" the same pronunciation as a noun and a verb, and
it read "10:45 AM" or "12 November 2026" digit by digit through g2p_en's generic
number expansion. This front-end:

  1. expands times (10:45 AM), day + month (12 November), years (2026) and other numbers
  2. tags parts of speech and resolves heteronyms with g2p_en's homograph table
     (conduct/NN -> K AA1 N D AH0 K T, conduct/VB -> K AH0 N D AH1 K T)
  3. uses the same lexicon as the forced aligner for every other known word
  4. falls back to the g2p_en neural model for unknown words (e.g. "andaroo", "rhyth")

Text is otherwise used exactly as given (no spelling correction).
"""

import re

import inflect

_inflect = inflect.engine()

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*([AaPp])\.?\s?[Mm]\.?(?=\W|$))?")
_DAY_MONTH_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(%s)\b" % _MONTHS)
_MONTH_DAY_RE = re.compile(r"\b(%s)\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!\s*:)" % _MONTHS)
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_YEAR_RE = re.compile(r"\b(1[1-9]\d\d|20\d\d)\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# letter names, used to spell out "AM" / "PM"
LETTERS = {"A": ["EY1"], "M": ["EH1", "M"], "P": ["P", "IY1"]}


def _words(n):
    return re.sub(r"[-,]", " ", _inflect.number_to_words(n, andword=""))


def _time(m):
    h, mins, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if mins == 0:
        out = _words(h) + " o'clock" if not ampm else _words(h)
    elif mins < 10:
        out = _words(h) + " oh " + _words(mins)
    else:
        out = _words(h) + " " + _words(mins)
    if ampm:
        out += " <{}> <M>".format(ampm.upper())
    return out


def _year(m):
    y = int(m.group(1))
    if 2000 <= y <= 2009:
        return _words(y)
    hi, lo = divmod(y, 100)
    if lo == 0:
        return _words(hi) + " hundred"
    return _words(hi) + (" oh " if lo < 10 else " ") + _words(lo)


def _ordinal(n):
    return re.sub(r"[-,]", " ", _inflect.ordinal(_inflect.number_to_words(n)))


def normalize(text):
    text = _TIME_RE.sub(_time, text)
    text = _DAY_MONTH_RE.sub(lambda m: _ordinal(int(m.group(1))) + " " + m.group(2), text)
    text = _MONTH_DAY_RE.sub(lambda m: m.group(1) + " " + _ordinal(int(m.group(2))), text)
    text = _ORDINAL_RE.sub(lambda m: _ordinal(int(m.group(1))), text)
    text = _YEAR_RE.sub(_year, text)
    text = _NUMBER_RE.sub(lambda m: _words(m.group(0)), text)
    return re.sub(r"\s+", " ", text).strip()


def read_lexicon(path):
    lexicon = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1 and parts[0].lower() not in lexicon:
                lexicon[parts[0].lower()] = parts[1:]
    return lexicon


class EnglishFrontend:
    def __init__(self, lexicon_path):
        import nltk
        from g2p_en import G2p

        for res, path in [
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("punkt_tab", "tokenizers/punkt_tab"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(res, quiet=True)
        self.lexicon = read_lexicon(lexicon_path)
        self.g2p = G2p()
        self.pos_tag = nltk.pos_tag

    def word_phones(self, word, pos):
        w = word.lower()
        if w in self.g2p.homograph2features:
            pron1, pron2, pos1 = self.g2p.homograph2features[w]
            return list(pron1 if pos.startswith(pos1) else pron2), "homograph"
        if w in self.lexicon:
            return list(self.lexicon[w]), "lexicon"
        if w in self.g2p.cmu:
            return list(self.g2p.cmu[w][0]), "cmudict"
        return list(self.g2p.predict(w)), "g2p"

    def text_to_words(self, text, verbose=True):
        """Sentence -> [(word, phones)] with ("sp", ["sp"]) at punctuation."""
        norm = normalize(text)
        tokens = re.findall(r"<[APM]>|[A-Za-z]+(?:'[A-Za-z]+)?|[,.;:!?]", norm)
        plain = [t for t in tokens if t[0] != "<" and t[0].isalpha()]
        tags = iter(self.pos_tag(plain))

        out, notes = [], []
        for t in tokens:
            if t.startswith("<"):
                out.append((t[1], LETTERS[t[1]]))
                notes.append("{}=letter".format(t[1]))
            elif t[0].isalpha():
                word, pos = next(tags)
                phones, source = self.word_phones(word, pos)
                out.append((word, phones))
                if source != "lexicon":
                    notes.append("{}/{}={}".format(word, pos, source))
            elif out and out[-1][0] != "sp":
                out.append(("sp", ["sp"]))
        while out and out[-1][0] == "sp":
            out.pop()
        if verbose:
            print("    normalised: " + norm)
            if notes:
                print("    " + ", ".join(notes))
        return out
