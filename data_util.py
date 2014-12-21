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

"""StylometricClustering - data util module
-----------------------------------------------------------

Note:
  data processing utilites: Tokenization,
  feature reduction, IO
-----------------------------------------------------------
"""
from __future__ import division
from collections import deque
from lxml import etree
from sklearn.decomposition import PCA
import codecs
import nltk
import numpy
import os
import sklearn

# PunktSentence Tokenizer
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path += [nltk_data_path,]
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
abbreviations = set(['e.g', 'i.e', 'b.c', 'a.d', 'ca', 'b.s', 'etc', 'esp', 'cf', 'chr', 'f.o.t', 'stat', 'f.o.c', 'b.sc', 'm.sc'])
# print sent_tokenizer._params.abbrev_types
sent_tokenizer._params.abbrev_types.update(abbreviations)

"""Tokenization"""
def return_tokenized_text(text):
    """Return sentence-tokenized text: one sentence per line."""
    text = text.strip()
    new_text = """"""

    for paragraph in text.split("\n\n"):
        new_text += "\n".join(tokenize_sentences(paragraph))
        new_text += "\n\n"
    return new_text.strip()

def tokenize_sentences(para):
    """Return sentences of a paragraph: one sentence per line."""
    para = para.strip()
    para = para.replace("\n", " ")
    sentences = sent_tokenizer.tokenize(para, realign_boundaries=True)
    return sentences


"""IO"""
def export_arff(matrix, attribute_list, file_path, file_name=None, labels_per_window=None, file_info=None):
    """Exports feature matrix to arff file"""
    with codecs.open(file_path, 'w', 'UTF-8') as new_file:
        # HEADER
        if file_info is not None:
            new_file.write('\n'.join(map(lambda line: "% " + line, file_info)))
            new_file.write("\n\n")

        new_file.write("@RELATION '"+str(file_name)+"'\n\n")

        new_file.write('\n'.join(map(lambda attribute: '@ATTRIBUTE "' + attribute + '" NUMERIC', attribute_list)))

        if labels_per_window is not None:
            labels = set(labels_per_window)
            new_file.write("\n@ATTRIBUTE class {"+','.join(labels)+"}")

        new_file.write("\n\n")

        # DATA
        new_file.write("@DATA\n")

        i = 0
        for row in matrix:
            new_file.write(','.join(map(str, row)))
            if labels_per_window is not None:
                new_file.write(','+labels_per_window[i])
                i += 1
            new_file.write('\n')

def create_text_fromXML(xml_filepath, txt_filepath):
    """Read xml-testfile and write text to extracted to given txt_filepath."""
    with codecs.open(xml_filepath, 'r', 'UTF-8') as xml_file:
        xml_tree = etree.parse(xml_file)
    
    with codecs.open(txt_filepath, 'w', 'UTF-8') as newFile:
        first_entry = True
        for entry in xml_tree.getroot():
            if entry.text is not None:
                if not first_entry:
                    newFile.write("\n\n")
                else:
                    first_entry = False
                newFile.write(entry.text)

def files_already_exist(list_of_filepaths):
    """Check if features are already exported."""
    for fpath in list_of_filepaths:
        if not os.path.exists(fpath):
            return False
    return True


"""Feature extraction and reduction"""
def normalize(feature_matrix):
    """Return normalized data."""
    X = numpy.array(feature_matrix)
    sklearn.preprocessing.normalize(X, norm='l2', axis=0, copy=False)
    return X

def pca(feature_matrix, feature_names, max_components=None):
    """Return PCA transformed data and corresponding labels."""
    pca = PCA(0.95)
    transformed = pca.fit_transform(feature_matrix)

    if max_components is not None and transformed.shape[1] > max_components:
        pca = PCA(max_components)
        transformed = pca.fit_transform(feature_matrix)

    labels = attribute_labels(pca.components_, feature_names)
    return transformed, labels

def attribute_labels(components, component_names, countVar=False):
    """Return list of attribute labels transformed by pca."""
    labels = []
    if components is None:
        return None

    i = 1
    num_dots = "..."
    for component in components:
        zipped = zip(component, component_names)
        zipped.sort(key=lambda x: abs(x[0]), reverse=True)
        if countVar:
            num_dots = "."*i
        labels.append(''.join(['{0:+.3f}'.format(tupel[0])+tupel[1] for tupel in zipped[0:5] if tupel[0] != 0]) + num_dots)
        i += 1
    return labels

def get_boundaries(paras):
    """Return paragraph boundaries in form of a deque of line numbers."""
    de = deque()
    boundary = -1
    for p in paras:
        boundary += len(p)
        de.append(boundary)
    return de

