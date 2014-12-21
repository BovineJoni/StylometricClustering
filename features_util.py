# Stylometric Clustering, Copyright 2014 Daniel Schneider.
# schneider.dnl(at)gmail.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
"""Stylometric Clustering - feature util module
-----------------------------------------------------------

Note:
  feature extraction utlities: load function words, remove
  punctuation/digits, tag sentence length etc.
-----------------------------------------------------------
"""
import string
import re

_DIGITWORDS_100 = set(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])
_FRACTION_WORDS = set(["half", "third", "fourth", "quarter", "fifth", "sixth", "seventh", "eighth", "ninth",
                "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth",
                "eightieth", "ninetieth", "halves", "thirds", "fourths", "quarters", "fifths", "sixths", "sevenths",
                "eighths", "ninths", "tenths", "elevenths", "twelfths", "thirteenths", "fourteenths", "fifteenths",
                "sixteenths", "seventeenths", "eighteenths", "nineteenths", "twentieths", "thirtieths", "fortieths",
                "fiftieths", "sixtieths", "seventieths", "eightieths", "ninetieths"])
_ORDINAL_WORDS = set(["zeroth", "noughth", "first", "second"])
_FRAC_ORD_WORDS = _FRACTION_WORDS.union(_ORDINAL_WORDS)

_ORDINAL = re.compile(r"[0-9]+(th|st|nd|rd)")

def remove_punctuation(sentence):
    """Remove punctuation tokens from sentence."""
    return filter(lambda symbol: symbol not in string.punctuation, sentence)

def remove_digits(sentence):
    """Remove digits in tokens."""
    return filter(lambda word: len(word) > 0, [w.strip("0123456789") for w in sentence])

def tag_sent_len(length):
    """Tag sentence length, s=short [0-5], m=medium [6-19], l=long [20-30], xl=xlong > 30"""
    if length < 6:
        return "s"
    elif length < 20:
        return "m"
    elif length < 31:
        return "l"
    else:
        return "xl"

def fill_function_words():
    """Read function words from file and return resulting set."""
    function_words = set()
    f = file('function_words.txt')

    for line in f:
        splitted = line.split()
        if len(splitted) == 1:
            function_words.add(splitted[0])
        else:
            function_words.add(tuple(splitted))
    return function_words

def startswith_digitword(word):
    """Return true if word starts with a digit word (e.g. twenty, thirty etc.)."""
    for digword in _DIGITWORDS_100:
        if word.startswith(digword):
            return True
    return False

def endswith_fractionword(word):
    """Return true if word ends with a fraction word."""
    for fracword in _FRAC_ORD_WORDS:
        if word.endswith(fracword):
            return True
    return False

def is_ordinal(word):
    """Return true if word is an ordinal (e.g. 1st, 2nd etc.)."""
    if _ORDINAL.match(word):
        return True
    else:
        return False