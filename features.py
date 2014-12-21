# -*- coding: utf-8 -*-

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

"""StylometricClustering - features module
-----------------------------------------------------------

Note:
  Feature extraction implementations divided into
  lexical and syntactical features:

  lexical:
    word-level, sentence-level, character-level, vocabulary
  syntactical:
    POS, verbs, punctuation
-----------------------------------------------------------
"""
from __future__ import division
from collections import Counter
from itertools import groupby
from nltk.stem import PorterStemmer
import nltk
import features_util as f_utils
import string

_FUNCTION_WORDS = f_utils.fill_function_words()
_FW_UNIGRAMS = set([word for word in _FUNCTION_WORDS if type(word) is not type(tuple())])
_FW_BIGRAMS = set([word for word in _FUNCTION_WORDS if type(word) is type(tuple()) and len(word) == 2])
_FW_TRIGRAMS = set([word for word in _FUNCTION_WORDS if type(word) is type(tuple()) and len(word) == 3])
_FW_4GRAMS = set([word for word in _FUNCTION_WORDS if type(word) is type(tuple()) and len(word) == 4])

_VERBS_TOBE = set(["am", "are", "is", "was", "were", "be", "been", "being", "aren't", "isn't", "wasn't", "weren't"])
_VERBS_TODO = set(["do", "does", "did", "done", "doing", "don't", "doesn't", "didn't"])
_VERBS_TOHAVE = set(["have", "has", "had", "having", "haven't", "hasn't", "hadn't"])

_VERBS_AUX = {"can": set(["can", "can't"]),
            "could": set(["could", "couldn't"]),
            "may": set(["may", "mayn't"]),
            "might": set(["might", "mightn't"]),
            "must": set(["must", "mustn't"]),
            "shall": set(["shall", "shan't"]),
            "should": set(["should", "shouldn't"]),
            "will": set(["will", "won't"]),
            "would": set(["would", "wouldn't"]),
            "ought": set(["ought", "oughtn't"]),
            "dare": set(["dare", "dares", "dared", "daren't"]),
            "need": set(["need", "needs", "needed", "needn't"]),
            "had_better": set(["had better"]),
            "used_to": set(["used to"])}

_DIGITWORDS_19 = set(["zero", "nought", "one", "two", "three", "four",
                "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen"])
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

_NUMTIMES_BIGRAMS = set([("one", "time"), ("two", "times"), ("three", "times")])
_MULTIPLICAT_NUM = set(["once", "twice", "thrice"])

"""feature_list (lexical & syntactical features)"""
paragraph_ind_features = ['total_words', 'mean_word_len', 'word_len_freq', 'short_words', 'medium_words', 'long_words', 'word_len_bigrams', 'word_len_trigrams', 'word_len_4grams',
    'mean_sent_len', 'mean_sent_len_char', 'short_sents', 'medium_sents', 'long_sents', 'extra_long_sents',
    'total_chars', 'char_freq', 'char_bigrams', 'char_bigrams_extended', 'char_trigrams', 'char_4grams', 'freq_consonants', 'vowels', 'digit_words_19', 'digit_words_20_99', 'fraction_words', 'ordinals', 'num_times', 'multiplicat_num', 'percent', 'digits', 'single_digits', 'two_digits', 'three_digits', 'alphas', 'uppers',
    'voc_yule', 'voc_richness', 'voc_hapax_legomenon', 'voc_dis_legomenon',
    'function_word', 'pos_freq', 'pos_bigrams', 'pos_trigrams', 'pos_4grams',
    'primary_verbs', 'auxiliary_verbs',
    'punctuation_ratio', 'common_punctuation_ratio', 'commas', 'periods', 'exclamation_marks', 'question_marks', 'colons', 'semi_colons', 'quotes', 'apostrophes']
paragraph_dep_features = ['sent_len_bigrams', 'sent_len_trigrams', 'sent_len_4grams']

"""Lexical features:
    
    word-level: total_words, mean_word_len, word_len_freq, short_words, medium_words, long_words, word_len_bigrams, word_len_trigrams, word_len_4grams
    sentence-level: mean_sent_len, mean_sent_len_char, short_sents, medium_sents, long_sents, extra_long_sents, sent_len_bigrams, sent_len_trigrams, sent_len_4grams
    character-level: total_chars, char_freq, char_bigrams, char_bigrams_extended, char_trigrams, char_4grams, freq_consonants, vowels, digit_words_19, digit_words_20_99, fraction_words, ordinals, num_times, multiplicat_num, percent, digits, single_digits, two_digits, three_digits, alphas, uppers
    vocabulary: voc_yule, voc_richness, voc_hapax_legomenon, voc_dis_legomenon
"""
"""Syntactical features:

    POS: function_word, pos_freq, pos_bigrams, pos_trigrams, pos_4grams 
    verbs: primary_verbs, auxiliary_verbs
    punctuation: punctuation_ratio, commas, periods, exclamation_marks, colons, semi_colons, quotes, apostrophes
"""


"""word_level"""
def total_words(sentences):
    """Return number of words."""
    num_words = map(lambda sentence: len(f_utils.remove_punctuation(sentence)), sentences)
    return float(sum(num_words))

def mean_word_len(sentences):
    """Return mean word length."""
    wordlen = []
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        wordlen += map(lambda word: len(word), s)
    
    if len(wordlen) == 0:
        return 0
    return sum(wordlen) / len(wordlen)

def word_len_freq(sentences):
    """Return relative frequency of 1-20 letter words."""
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        wordlen = map(lambda word: len(word), s)
        words_shorter20 = filter(lambda wordlen: wordlen <= 20, wordlen)
        counter += Counter(map(lambda word: str(word), words_shorter20))
    
    sum_words = sum(counter.values())
    for key in counter.iterkeys():
        yield "word_len_freq_" + key, counter[key] / sum_words

def short_words(sentences):
    """Return short word ratio."""
    shortwords = []
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        shortwords += filter(lambda word: len(word) <= 3, s)
        total_words += len(s)
    return len(shortwords) / total_words

def medium_words(sentences):
    """Return medium long word ratio."""
    mediumwords = []
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        mediumwords += filter(lambda word: len(word) in range(3,8), s)
        total_words += len(s)
    return len(mediumwords) / total_words

def long_words(sentences):
    """Return long word ratio."""
    longwords = []
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        longwords += filter(lambda word: len(word) > 7, s)
        total_words += len(s)
    return len(longwords) / total_words

def word_len_bigrams(sentences):
    """Return word length bigram frequencys."""
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        wordlen = map(lambda word: len(word), s)
        bigrams = nltk.util.ngrams(wordlen, 2)
        bigrams = [str(a) +'_'+ str(b) for a, b in bigrams]
        counter += Counter(bigrams)

    sum_bigrams = sum(counter.values())
    for key in counter.iterkeys():
        yield "word_len_bigram_" + key, counter[key] / sum_bigrams

def word_len_trigrams(sentences):
    """Return word length trigram frequencys."""
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        wordlen = map(lambda word: len(word), s)
        trigrams = nltk.util.ngrams(wordlen, 3)
        trigrams = [str(a) +'_'+ str(b) + '_' + str(c) for a, b, c in trigrams]
        counter += Counter(trigrams)

    sum_trigrams = sum(counter.values())
    for key in counter.iterkeys():
        yield "word_len_trigram_" + key, counter[key] / sum_trigrams

def word_len_4grams(sentences):
    """Return word length 4-gram frequecys."""
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        wordlen = map(lambda word: len(word), s)
        fourgram = nltk.util.ngrams(wordlen, 4)
        fourgram = [str(a) +'_'+ str(b) + '_' + str(c) + '_' + str(d) for a, b, c, d in fourgram]
        counter += Counter(fourgram)

    sum_4grams = sum(counter.values())
    for key in counter.iterkeys():
        yield "word_len_4gram_" + key, counter[key] / sum_4grams


"""sentence_length"""
def mean_sent_len(sentences):
    """Return mean sentence length."""
    sentlen = map(lambda sentence: len(f_utils.remove_punctuation(sentence)), sentences)
    return sum(sentlen) / len(sentlen)

def mean_sent_len_char(sentences):
    """Return mean sentence length in characters."""
    sentlen_char = map(lambda sentence: sum(map(lambda word: len(word), f_utils.remove_punctuation(sentence))), sentences)
    return sum(sentlen_char) / len(sentlen_char)
    
def short_sents(sentences):
    """Return short sentence ratio."""
    shortsents = filter(lambda sentence: len(f_utils.remove_punctuation(sentence)) <= 5, sentences)
    return len(shortsents) / len(sentences)

def medium_sents(sentences):
    """Return medium sentence ratio."""
    mediumsents = filter(lambda sentence: len(f_utils.remove_punctuation(sentence)) in range(6,20), sentences)
    return len(mediumsents) / len(sentences)

def long_sents(sentences):
    """Return long sentence ratio."""
    longsents = filter(lambda sentence: len(f_utils.remove_punctuation(sentence)) in range(20,31), sentences)
    return len(longsents) / len(sentences)

def extra_long_sents(sentences):
    """Return extra long sentence ratio."""
    xlongsents = filter(lambda sentence: len(f_utils.remove_punctuation(sentence)) >= 31, sentences)
    return len(xlongsents) / len(sentences)

def sent_len_bigrams(sentences):
    """Return sentence length bigram frequencys."""
    sent_len = map(lambda sentence: f_utils.tag_sent_len(len(f_utils.remove_punctuation(sentence))), sentences)
    bigrams = nltk.util.ngrams(sent_len, 2)
    bigrams = [a + b for a, b in bigrams]
    counter = Counter(bigrams)

    sum_counter = sum(counter.values())

    for key in counter.iterkeys():
        yield "sent_len_bigram_" + key, counter[key] / sum_counter

def sent_len_trigrams(sentences):
    """Return sentence length trigram frequencys."""
    sent_len = map(lambda sentence: f_utils.tag_sent_len(len(f_utils.remove_punctuation(sentence))), sentences)
    trigram = nltk.util.ngrams(sent_len, 3)
    trigram = [a + b + c for a, b, c in trigram]
    counter = Counter(trigram)

    sum_counter = sum(counter.values())

    for key in counter.iterkeys():
        yield "sent_len_trigram_" + key, counter[key] / sum_counter

def sent_len_4grams(sentences):
    """Return sentence length 4-gram frequecys."""
    sent_len = map(lambda sentence: f_utils.tag_sent_len(len(f_utils.remove_punctuation(sentence))), sentences)
    fourgram = nltk.util.ngrams(sent_len, 4)
    fourgram = [a + b + c + d for a, b, c, d in fourgram]
    counter = Counter(fourgram)

    sum_counter = sum(counter.values())

    for key in counter.iterkeys():
        yield "sent_len_4gram_" + key, counter[key] / sum_counter


"""character"""
def total_chars(sentences):
    """Return number of characters."""
    num_chars = []
    for s in sentences:
        num_chars += map(lambda word: len(word), s)
    return float(sum(num_chars))

def char_freq(sentences):
    """Return character frequencies."""
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        for w in s:
            counter += Counter(w.lower())
    sum_char = sum(counter.values())
    for k in counter.iterkeys():
        yield "char_freq_" + k, counter[k] / sum_char

def char_bigrams(sentences):
    """Return character bigram frequencies."""
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        for w in s:
            bigrams = nltk.ngrams(w.lower(), 2)
            bigrams = [a +'_'+ b for a, b in bigrams]  
            counter += Counter(bigrams)  
    sum_counter = sum(counter.values())
    for k in counter.iterkeys():
        yield "char_bigram_" + k, counter[k] / sum_counter

def char_bigrams_extended(sentences):
    """Return character bigram frequencies + frequencies of most common bigrams."""
    comm_cc2grams = set([("t","h"), ("s","t"), ("n","d")])
    comm_vc2grams = set([("a","n"), ("i","n"), ("e","r"),
                    ("e","s"), ("o","n"), ("a","t"),
                    ("e","n"), ("o","r"), ("h","e")])
    comm_cv2grams = set([("r","e"), ("t","i")])
    comm_vv2grams = set([("e","a")])
    
    vowels = set("aeiou")
    consonants = set("qwrtzpsdfghjklyxcvbnm")
    bigrams = []

    for s in sentences:
        s = f_utils.remove_punctuation(s)
        for w in s:
            bigrams += nltk.ngrams(w.lower(), 2)
            
    aa_bigrams = filter(lambda bigram: bigram[0].isalpha() and bigram[1].isalpha(), bigrams)
    vv_bigrams = filter(lambda bigram: bigram[0] in vowels and bigram[1] in vowels, aa_bigrams)
    vc_bigrams = filter(lambda bigram: bigram[0] in vowels and bigram[1] in consonants, aa_bigrams)
    cv_bigrams = filter(lambda bigram: bigram[0] in consonants and bigram[1] in vowels, aa_bigrams)
    cc_bigrams = filter(lambda bigram: bigram[0] in consonants and bigram[1] in consonants, aa_bigrams)
    
    common_bigrams = {}
    
    filtered_cc = Counter(filter(lambda bigram: bigram in comm_cc2grams, cc_bigrams))
    for bigram, num_times in filtered_cc.iteritems():
        common_bigrams[bigram[0]+"_"+bigram[1]] = num_times / len(cc_bigrams)
    
    filtered_vc = Counter(filter(lambda bigram: bigram in comm_vc2grams, vc_bigrams))
    for bigram, num_times in filtered_vc.iteritems():
        common_bigrams[bigram[0]+"_"+bigram[1]] = num_times / len(vc_bigrams)

    filtered_cv = Counter(filter(lambda bigram: bigram in comm_cv2grams, cv_bigrams))
    for bigram, num_times in filtered_cv.iteritems():
        common_bigrams[bigram[0]+"_"+bigram[1]] = num_times / len(cv_bigrams)
    
    filtered_vv = Counter(filter(lambda bigram: bigram in comm_vv2grams, vv_bigrams))
    for bigram, num_times in filtered_vv.iteritems():
        common_bigrams[bigram[0]+"_"+bigram[1]] = num_times / len(vv_bigrams)
    
    yield "char_aa_bigrams", len(aa_bigrams) / len(bigrams)
    yield "char_vv_bigrams", len(vv_bigrams) / len(aa_bigrams)
    yield "char_vc_bigrams", len(vc_bigrams) / len(aa_bigrams)
    yield "char_cv_bigrams", len(cv_bigrams) / len(aa_bigrams)
    yield "char_cc_bigrams", len(cc_bigrams) / len(aa_bigrams)
    for key, value in common_bigrams.iteritems():
        yield "char_bigram_" + key, value

def char_trigrams(sentences):
    """Return character trigram frequencies."""
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        for w in s:
            if len(w) <= 3 or not w.isalpha():
                continue
            trigram = nltk.ngrams(w.lower(), 3)
            trigram = [a +'_'+ b + '_' + c for a, b, c in trigram]  
            counter += Counter(trigram)  
    sum_counter = sum(counter.values())
    for k in counter.iterkeys():
        yield "char_trigram_" + k, counter[k] / sum_counter

def char_4grams(sentences):
    """Return character 4-gram frequencies."""
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        for w in s:
            fourgram = nltk.ngrams(w.lower(), 4)
            fourgram = [a +'_'+ b + '_' + c + '_' + d for a, b, c, d in fourgram]  
            counter += Counter(fourgram)  
    sum_counter = sum(counter.values())
    for k in counter.iterkeys():
        yield "char_4gram_" + k, counter[k] / sum_counter

def freq_consonants(sentences):
    """Return ratio of most frequent consonant groups / total alpha-chars."""
    alphas = []
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        for w in s:
            alphas += filter(lambda char: char.isalpha(), w)
    cons_tnsrh = filter(lambda cons: cons.lower() in "tnsrh", alphas)
    cons_ldcpf = filter(lambda cons: cons.lower() in "ldcpf", alphas)
    cons_mwybg = filter(lambda cons: cons.lower() in "mwybg", alphas)
    cons_jkqvxz = filter(lambda cons: cons.lower() in "jkqvxz", alphas)

    yield "freq_cons_tnsrh", len(cons_tnsrh) / len(alphas)
    yield "freq_cons_ldcpf", len(cons_ldcpf) / len(alphas)
    yield "freq_cons_mwybg", len(cons_mwybg) / len(alphas)
    yield "freq_cons_jkqvxz", len(cons_jkqvxz) / len(alphas)

def vowels(sentences):
    """Return ratio vowels / total alpha-chars."""
    alphas = []
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        for w in s:
            alphas += filter(lambda char: char.isalpha(), w)
    num_vowels = filter(lambda vow: vow.lower() in "aeiou", alphas)

    if len(alphas) == 0:
        return 0
    return len(num_vowels) / len(alphas)

def digit_words_19(sentences):
    """Return ratio digit words <= 19 / digits <= 19
        + single ratios / words."""
    counter_digwords = Counter()
    counter_digits = Counter()
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
        counter_digwords += Counter(filter(lambda word: word in _DIGITWORDS_19, s))
        counter_digits += Counter(filter(lambda word: word.isdigit() and float(word) in range(0,20), s))

    sum_digwords = sum(counter_digwords.values())
    sum_digits = sum(counter_digits.values())
    total = sum_digwords + sum_digits
    if total == 0:
        yield "digit_words_19", 0
    else:
        yield "digit_words_19", sum_digwords / total_words
        yield "digits_19", sum_digits / total_words

        for key in counter_digwords.iterkeys():
            yield "digit_words_19_" + key, counter_digwords[key] / total
        for key in counter_digits.iterkeys():
            yield "digits_19_" + key, counter_digits[key] / total

def digit_words_20_99(sentences):
    """Return ratio digit words in range (20,100) / digits in range (20,100)."""
    counter_digwords = Counter()
    counter_digits = Counter()
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
        counter_digwords += Counter(filter(lambda word: f_utils.startswith_digitword(word), s))
        counter_digits += Counter(filter(lambda word: word.isdigit() and float(word) in range(20, 100), s))
    
    sum_digwords = sum(counter_digwords.values())
    sum_digits = sum(counter_digits.values())
    total = sum_digwords + sum_digits
    if total == 0:
        yield "digit_words_20_99", 0
    else:
        yield "digit_words_20_99", sum_digwords / total_words
        yield "digits_20_99", sum_digits / total_words



def fraction_words(sentences):
    """Return ratio fraction words / total words."""
    counter = Counter()
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
        counter += Counter(filter(lambda word: f_utils.endswith_fractionword(word), s))

    sum_fraction_words = sum(counter.values())
    if sum_fraction_words == 0:
        return 0
    else:
        return sum_fraction_words / total_words

def ordinals(sentences):
    """Return ratio ordinals in form dd+['th', 'st', 'nd', 'rd'] / total words."""
    counter = Counter()
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
        counter += Counter(filter(lambda word: f_utils.is_ordinal(word), s))

    sum_ordinals = sum(counter.values())
    if sum_ordinals == 0:
        return 0
    else:
        return sum_ordinals / total_words

def num_times(sentences):
    """Return ratio of one/two/three time(s) / total words."""
    counter = Counter()
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
        s = map(lambda word: word.lower(), s)
        bigrams = nltk.ngrams(s, 2)  
        counter += Counter(filter(lambda bigram: bigram in _NUMTIMES_BIGRAMS, bigrams))
    
    sum_counter = sum(counter.values())
    if sum_counter == 0:
        yield "times_" + "one_time", 0
    else:
        for key in counter.iterkeys():
            yield "times_" + key[0] + "_" + key[1], counter[key] / total_words

def multiplicat_num(sentences):
    """Return ratio of multiplicative numbers (once, twice, thrice) / total words."""
    counter = Counter()
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
        counter += Counter(filter(lambda word: word.lower() in _MULTIPLICAT_NUM, s))

    sum_counter = sum(counter.values())
    if sum_counter == 0:
        yield "times_" + "once", 0
    else:
        for key in counter.iterkeys():
            yield "times_" + key, counter[key] / total_words

def percent(sentences):
    """Return ratio of word 'percent' & '%' / total words."""
    counter = Counter()
    total_words = 0
    for s in sentences:
        counter += Counter(filter(lambda word: word.lower()=="percent" or word == "%", s))
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
    sum_counter = sum(counter.values())
    if sum_counter == 0:
        yield "percent", 0
    else:
        for key in counter.iterkeys():
            yield key, counter[key] / total_words


def digits(sentences):
    """Return ratio digits / total characters."""
    digits = 0
    total = 0
    for s in sentences:
        for word in s:
            for c in word:
                if c.isdigit():
                    digits += 1
                total += 1
    return digits / total

def single_digits(sentences):
    """Return ratio single digit numbers / total digits."""
    digits = []
    for s in sentences:
        digits += filter(lambda digit: digit.isdigit(), s)

    single_digits = filter(lambda digit: len(digit) == 1, digits)

    if len(digits) == 0:
        return "digits_single_digits", 0
    return "digits_single_digits", len(single_digits) / len(digits)

def two_digits(sentences):
    """Return ratio 2-digits numbers / total digits."""
    digits = []
    for s in sentences:
        digits += filter(lambda digit: digit.isdigit(), s)

    two_digits = filter(lambda digit: len(digit) == 2, digits)

    if len(digits) == 0:
        return "digits_two_digits", 0
    return "digits_two_digits", len(two_digits) / len(digits)


def three_digits(sentences):
    """Return ratio 3-digits numbers / total digits."""
    digits = []
    for s in sentences:
        digits += filter(lambda digit: digit.isdigit(), s)

    three_digits = filter(lambda digit: len(digit) == 3, digits)
    
    if len(digits) == 0:
        return "digits_three_digits", 0
    return "digits_three_digits", len(three_digits) / len(digits)

def alphas(sentences):
    """Return ratio alphas / total characters."""
    alphas = 0
    total = 0
    for s in sentences:
        for word in s:
            for c in word:
                if c.isalpha():
                    alphas +=1
                total += 1
    return alphas / total

def uppers(sentences):
    """Return ratio uppercase chars / total characters."""
    uppers = 0
    total = 0
    for s in sentences:
        for word in s:
            for c in word:
                if c.isupper():
                    uppers += 1
                if c.isalpha():
                    total += 1
    return uppers / total


"""vocabulary"""
def voc_yule(sentences):
    """Return modified yule's I measure."""
    stemmer = PorterStemmer()
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        s = f_utils.remove_digits(s)
        counter += Counter(map(lambda word: stemmer.stem(word).lower(), s))
 
    M1 = float(len(counter))
    # print "M1 {}".format(M1)
    M2 = sum([len(list(g))*(freq**2) for freq,g in groupby(counter.values())])
 
    try:
        return ((M1*M2)/(M2-M1))/sum(counter.values())
    except ZeroDivisionError:
        return 0.0

def voc_richness(sentences):
    """Return ratio different words / total words."""
    counter = Counter()
    stemmer = PorterStemmer()

    for s in sentences:
        s = f_utils.remove_punctuation(s)
        s = f_utils.remove_digits(s)
        counter += Counter((map(lambda word: stemmer.stem(word).lower(), s)))
    return len(counter) / sum(counter.values())

def voc_hapax_legomenon(sentences):
    """Return ratio unique words / total words."""
    counter = Counter()
    stemmer = PorterStemmer()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        s = f_utils.remove_digits(s)
        counter += Counter(map(lambda word: stemmer.stem(word).lower(), s))

    unique = filter(lambda word: counter[word] == 1, counter)
    sum_counter = sum(counter.values())
    # print "unique {} / total {}".format(len(unique), sum_counter)
    return len(unique) / sum_counter

def voc_dis_legomenon(sentences):
    """Return ratio words occuring twice / total words."""
    counter = Counter()
    stemmer = PorterStemmer()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        s = f_utils.remove_digits(s)
        counter += Counter(map(lambda word: stemmer.stem(word).lower(), s))

    twice = filter(lambda word: counter[word] == 2, counter)
    sum_counter = sum(counter.values())
    # print "twice {} / total {}".format(len(twice), sum_counter)
    return len(twice) / sum_counter


def voc_bottom10(sentences):
    """Return ratio least frequent words (bottom 10%) / total words."""
    counter = Counter()
    stemmer = PorterStemmer()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        s = f_utils.remove_digits(s)
        counter += Counter(map(lambda word: stemmer.stem(word).lower(), s))

    # least common 10% of occuring words
    bottom_10 = max(int(round(len(counter)*0.1)), 1)
    least_common_10 = counter.most_common()[:-bottom_10-1:-1]
    # sum_least_common = sum(least_common_10.values())
    # print "least_common_10"
    # print least_common_10
    sum_least_common = sum([lc[1] for lc in least_common_10])

    sum_counter = sum(counter.values())
    # print "least common {} / total {}".format(sum_least_common, sum_counter)
    return sum_least_common / sum_counter    



"""Syntactical features:

    POS: function_word, pos_freq, pos_bigrams, pos_trigrams, pos_4grams
    verbs: primary_verbs, auxiliary_verbs 
    punctuation: punctuation_ratio, common_punctuation_ratio, commas, periods, exclamation_marks, question_marks, colons, semi_colons, quotes, apostrophes
"""

"""POS"""
def function_word(sentences):
    """Return function word frequencies."""
    counter = Counter()
    features = []
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        lowered = map(lambda word: word.lower(), s)

        # check if word is in _FW_UNIGRAMS
        for i, word in enumerate(lowered):
            if word in _FW_UNIGRAMS:
                # check if word is part of bi/tri/4-gram
                try:
                    if not set(nltk.ngrams(lowered[max(i-1, 0):min(i+2, len(lowered)-1)], 2)).isdisjoint(_FW_BIGRAMS):
                        pass
                    elif not set(nltk.ngrams(lowered[max(i-2, 0):min(i+3, len(lowered)-1)], 3)).isdisjoint(_FW_TRIGRAMS):
                        pass
                    elif not set(nltk.ngrams(lowered[max(i-3, 0):min(i+4, len(lowered)-1)], 4)).isdisjoint(_FW_4GRAMS):
                        pass
                    else:
                        # print "UNIGRAM_FW found: {}".format(word)
                        features.append(word)
                except IndexError:
                    print "Error while extracting function words."

        bigrams = nltk.util.ngrams(lowered, 2)
        for i, bigram in enumerate(bigrams):
            if bigram in _FW_BIGRAMS:
                # check if bigram is part of tri/4-gram
                try:
                    if not set(nltk.ngrams(lowered[max(0, i-2): min(i+3, len(lowered)-1)], 3)).isdisjoint(_FW_TRIGRAMS):
                        pass
                    elif not set(nltk.ngrams(lowered[max(0, i-3): min(i+4, len(lowered)-1)], 4)).isdisjoint(_FW_4GRAMS):
                        pass
                    else:
                        # print "BIGRAM_FW found: {}".format(bigram)
                        features.append(bigram[0] +"_"+ bigram[1])
                except IndexError:
                    print "Error while extracting function words."

        trigrams = nltk.util.ngrams(lowered, 3)
        for i, trigram in enumerate(trigrams):
            if trigram in _FW_TRIGRAMS:
                # check if trigram is part of 4-gram
                try:
                    if not set(nltk.ngrams(lowered[max(0, i-3): min(i+4, len(lowered)-1)], 4)).isdisjoint(_FW_4GRAMS):
                        pass
                    else:
                        # print "TRIGRAM_FW found: {}".format(trigram)
                        features.append(trigram[0] +"_"+ trigram[1] +"_"+ trigram[2])

                except IndexError:
                    print "Error while extracting function words."

        fourgrams = nltk.util.ngrams(lowered, 4)    
        for fourgram in fourgrams:
            if fourgram in _FW_4GRAMS:
                # print "4GRAM_FW found: {}".format(fourgram)
                features.append(fourgram[0] +"_"+ fourgram[1] +"_"+ fourgram[2] +"_"+ fourgram[3])
    counter = Counter(features)
    sum_counter = sum(counter.values())
    for key in counter.iterkeys():
        yield "fw_" + key, counter[key] / sum_counter


def pos_freq(sentences):
    """Return POS tag frequencies."""
    counter = Counter()
    for s in sentences:
        tagged = nltk.pos_tag(s)
        tags = [tag[1] for tag in tagged]
        counter += Counter(tags)
    sum_counter = sum(counter.values())
    for key in counter.iterkeys():
        yield "pos_freq_" + key, counter[key] / sum_counter


def pos_bigrams(sentences):
    """Return POS bigram frequencies."""
    counter = Counter()
    for s in sentences:
        tagged = nltk.pos_tag(s)
        tags = [tag[1] for tag in tagged]
        bigram = nltk.util.ngrams(tags, 2)
        bigram = [a +'_'+ b for a, b in bigram]
        counter += Counter(bigram)
    sum_counter = sum(counter.values())
    for key in counter.iterkeys():
        yield "pos_bigram_" + key, counter[key] / sum_counter

def pos_trigrams(sentences):
    """Return POS trigram frequencies."""
    counter = Counter()
    for s in sentences:
        tagged = nltk.pos_tag(s)
        tags = [tag[1] for tag in tagged]
        trigram = nltk.util.ngrams(tags, 3)
        trigram = [a +'_'+ b + '_' + c for a, b, c in trigram]
        counter += Counter(trigram)
    sum_counter = sum(counter.values())
    for key in counter.iterkeys():
        yield "pos_trigram_" + key, counter[key] / sum_counter

def pos_4grams(sentences):
    """Return POS 4-gram frequencies."""
    counter = Counter()
    for s in sentences:
        tagged = nltk.pos_tag(s)
        tags = [tag[1] for tag in tagged]
        fourgram = nltk.util.ngrams(tags, 4)
        fourgram = [a +'_'+ b + '_' + c + '_' + d for a, b, c, d in fourgram]
        counter += Counter(fourgram)
    sum_counter = sum(counter.values())
    for key in counter.iterkeys():
        yield "pos_4gram_" + key, counter[key] / sum_counter



"""verbs"""
def primary_verbs(sentences):
    """Return ratio primary verb / all primary verbs

    + ratio all primary verbs / total words.
    """
    total_words = 0
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
        counter += Counter({"verb_tobe": len(filter(lambda word: word in _VERBS_TOBE, s))})
        counter += Counter({"verb_todo": len(filter(lambda word: word in _VERBS_TODO, s))})
        counter += Counter({"verb_tohave": len(filter(lambda word: word in _VERBS_TOHAVE, s))})
    sum_counter = sum(counter.values())
    for key in counter.iterkeys():
        yield key, counter[key] / sum_counter
    yield "primary_verbs", sum_counter / total_words

def auxiliary_verbs(sentences):
    """Return ratio auxiliary verb / all auxiliary verbs 
        
    + ratio all auxiliary verbs / total words.
    """
    total_words = 0
    counter = Counter()
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        total_words += len(s)
        for key in _VERBS_AUX.iterkeys():
            counter += Counter({key: len(filter(lambda word: word in _VERBS_AUX[key], s))})
    sum_counter = sum(counter.values())
    for key in counter.iterkeys():
        yield "auxverb_" + key, counter[key] / sum_counter
    yield "auxiliary_verbs", sum_counter / total_words




"""punctuation"""
def punctuation_ratio(sentences):
    """Return ratio punctuation / total characters."""
    total_chars = 0
    punct_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
        total_chars += sum(map(lambda token: len(token), s))
    return len(punct_only) / total_chars

def common_punctuation_ratio(sentences):
    """Return ratio common punctuation / total punctuation symbols."""
    punct_only = []
    common_punct_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
    common_punct_only = filter(lambda common_punct: common_punct in ".,;:?!\"'", punct_only)
    try:
        return len(common_punct_only) / len(punct_only)
    except ZeroDivisionError:
        return 0.0
    
def commas(sentences):
    """Return ratio commas / total punctuation symbols."""
    punct_only = []
    commas_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
    commas_only = filter(lambda punct_symbol: punct_symbol == ",", punct_only)
    try:
        return len(commas_only) / len(punct_only)
    except ZeroDivisionError:
        return 0.0

def periods(sentences):
    """Return ratio periods / total punctuation symbols."""
    punct_only = []
    periods_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
    periods_only = filter(lambda punct_symbol: punct_symbol == ".", punct_only)
    try:
        return len(periods_only) / len(punct_only)
    except ZeroDivisionError:
        return 0.0

def exclamation_marks(sentences):
    """Return ratio exlamation marks / total punctuation symbols."""
    punct_only = []
    excl_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
    excl_only = filter(lambda punct_symbol: punct_symbol == "!", punct_only)
    try:
        return len(excl_only) / len(punct_only)
    except ZeroDivisionError:
        return 0.0

def question_marks(sentences):
    """Return ratio question marks / total punctuation symbols."""
    punct_only = []
    quest_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
    quest_only = filter(lambda punct_symbol: punct_symbol == "?", punct_only)
    try:
        return len(quest_only) / len(punct_only)
    except ZeroDivisionError:
        return 0.0
    
def colons(sentences):
    """Return ratio colons / total punctuation symbols."""
    punct_only = []
    colons_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
    colons_only = filter(lambda punct_symbol: punct_symbol == ":", punct_only)
    try:
        return len(colons_only) / len(punct_only)
    except ZeroDivisionError:
        return 0.0

def semi_colons(sentences):
    """Return ratio semi colons / total punctuation symbols."""
    punct_only = []
    semic_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
    semic_only = filter(lambda punct_symbol: punct_symbol == ";", punct_only)
    try:
        return len(semic_only) / len(punct_only)
    except ZeroDivisionError:
        return 0.0

def quotes(sentences):
    """Return ratio quotes / total punctuation symbols."""
    punct_only = []
    quotes_only = []
    for s in sentences:
        punct_only += filter(lambda token: token in string.punctuation, s)
    quotes_only = filter(lambda punct_symbol: punct_symbol == "\"", punct_only)
    try:
        return len(quotes_only) / len(punct_only)
    except ZeroDivisionError:
        return 0.0

def apostrophes(sentences):
    """Return ratio of words which contain apostrophes / total words."""
    words_with_apos = []
    total_words = 0
    for s in sentences:
        s = f_utils.remove_punctuation(s)
        words_with_apos += filter(lambda word: "'" in word, s)
        total_words += len(s)
    return len(words_with_apos) / total_words



def word_freq(sentences):
    """Return word frequencies."""
    counter = Counter()
    for s in sentences:
        lowered = map(lambda word: word.lower(), s)
        counter += Counter(lowered)
        # counter += Counter(s.lower())
    sum_word = sum(counter.values())
    for k in counter.iterkeys():
        yield "word_freq_" + k, counter[k] / sum_word

def word_bigrams(sentences):
    """Return dictionary of bigram frequencies."""
    counter = Counter()
    for s in sentences:
        lowered = map(lambda word: word.lower(), s)
        bigrams = nltk.bigrams(lowered)
        bigrams = [a +'_'+ b for a, b in bigrams]
        counter += Counter(bigrams)
    sum_bigrams = sum(counter.values())
    for k in counter.iterkeys():
        yield "word_bigram_" + k, counter[k] / sum_bigrams
