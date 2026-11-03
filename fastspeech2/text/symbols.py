""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

[modified] Mandarin pinyin symbols removed and the Telugu phone set added. Both languages
share one table: ARPAbet phones are upper-case, Telugu phones lower-case, so the English
model simply never uses the Telugu rows of the embedding (and vice versa).
"""

from text import cmudict, telugu

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_telugu = ["@" + s for s in telugu.PHONES]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _telugu
    + _silences
)
