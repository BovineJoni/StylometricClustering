# -*- coding: utf-8 -*-

# StylometricClustering, Copyright 2014 Daniel Schneider.
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

"""StylometricClustering - data module
-----------------------------------------------------------

Note:
  Data module for extracting features and exporting data.
  Dimensionality gets reduced by a PCA and arff- and
  numpy-files get exported to the subdirectory
  'extracted_features'.
-----------------------------------------------------------
"""
from __future__ import division
from collections import deque
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import LineTokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import DictVectorizer
from types import GeneratorType
import collections
import data_util as util
import itertools
import features
import gc
import nltk
import numpy
import pprint
import string
import time
import os
import logging
import pickle

_WINDOW_SIZE = 6
_STEP_SIZE = 1
_PARAIND_FEAT = ["function_word", "mean_sent_len"]
_PARADEP_FEAT = []
_DO_PCA = True

_NEW_SUBDIR = "extracted_features"
# _NEW_SUBDIR = None

class Data(object):
    """Data class - stores feature sets of given text."""

    def __init__(self, article_id):
        """Initialize a data object."""
        self.id = article_id
        self.data = {}

        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    def extract_data(self, filepath, ind_features=_PARAIND_FEAT, dep_features=_PARADEP_FEAT, labels_per_sent=None, labels_per_window=None):
        """Extract features, reduce dimensions with a PCA and return data.

        Exports raw- and PCA-reduced data both in arff- and numpy-format.
        """
        start = time.clock()
        self.dictVectorizer = DictVectorizer(sparse=False)
        filename = os.path.split(filepath)[1]
        directory = os.path.split(filepath)[0]
        plain_reader = PlaintextCorpusReader(
            directory, 
            [filename],
            word_tokenizer=RegexpTokenizer("(-?\d+\.\d+)|[\w']+|["+string.punctuation+"]"),
            sent_tokenizer=LineTokenizer(blanklines="discard"),
            encoding='utf8')

        # create new subdir for extracted data
        if _NEW_SUBDIR is not None:
            path = os.path.join(directory, _NEW_SUBDIR)
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, os.path.splitext(filename)[0])            
            # print "path {}".format(path)
        else:
            path = os.path.splitext(filepath)[0]
            # print "path {}".format(path)

        # filepaths for weka- and numpy-files
        arff_filepath = path + ".arff"
        arff_filepath_pca = path + "_pca95.arff"
        numpy_filepath = path + ".npy"
        numpy_filepath_pca = path + "_pca95.npy"
        
        # print(":time: Reader created, time elapsed {}").format(time.clock() - start)
        paras = plain_reader.paras()
        # print(":time: Paras created, time elapsed {}").format(time.clock() - start)
        sents = plain_reader.sents()
        # print(":time: Sents created, time elapsed {}").format(time.clock() - start)

        # get paragraph boundaries for sliding-window
        self.boundaries = util.get_boundaries(paras)
        boundaries_backup = self.boundaries

        # check if all files necessary exist, if yes - unpickle/load them and return data
        if util.files_already_exist([numpy_filepath_pca,]):
            print "Features already extracted. Calculating clusters...\n"
            matrix_sklearn_pca = numpy.load(numpy_filepath_pca)
            return filepath, self.boundaries, matrix_sklearn_pca, len(sents)

        # save correct target-labels and additional info of current data
        targets_path = open(path + ".tbs", "wb")
        pickle.dump((labels_per_sent, labels_per_window, boundaries_backup, len(sents), _WINDOW_SIZE, _STEP_SIZE), targets_path)

        # print(":time: Boundaries calculated, time elapsed {}").format(time.clock() - start)
        self.data = self.extract_features(sents, _WINDOW_SIZE, _STEP_SIZE, ind_features, dep_features)
        # self.data[year] = self.extract_features_para(paras, ind_features, dep_features)
        # print(":time: Features extracted, time elapsed {}").format(time.clock() - start)
        self.all_features = self.unified_features(self.data)
        # print(":time: Unified features, time elapsed {}").format(time.clock() - start)
        matrix_sklearn = self.feature_matrix_sklearn(self.generator_data(self.data))
        # print(":time: Matrix sklearn created, time elapsed {}").format(time.clock() - start)
        matrix_sklearn = util.normalize(matrix_sklearn)
        # print(":time: Matrix normalized, time elapsed {}").format(time.clock() - start)
        
        print "Exporting raw-data..."
        util.export_arff(matrix_sklearn, self.dictVectorizer.get_feature_names(), arff_filepath, filename+"_RAW", labels_per_window, file_info=None)
        numpy.save(numpy_filepath, matrix_sklearn)
        
        # print "matrix dimensions before pca: {}".format(matrix_sklearn.shape)
        feature_names, feature_names_part = None, None
        if _DO_PCA:
            print "PCA calculation..."
            matrix_sklearn_pca, feature_names = util.pca(matrix_sklearn, self.dictVectorizer.get_feature_names())
            util.export_arff(matrix_sklearn_pca, feature_names, arff_filepath_pca, filename+"_PCA95", labels_per_window, file_info=None)
            numpy.save(numpy_filepath_pca, matrix_sklearn_pca)
            
            del matrix_sklearn
        gc.collect()
        return filepath, boundaries_backup, matrix_sklearn_pca, len(sents)

    def add_feature(self, feature_set, feature_name, value, override=True):
        """Add feature name and value to the feature set of the current sliding-window.

        Distinguishes between single-value features and multi-valued features.
        """
        try:
            for v in value:
                if type(v) is not type(()):
                    # not a key, value pair of a generator, but a single value with custom feature_name
                    feature_name, value = value[0], value[1]
                    raise TypeError

                if not override and v[0] in feature_set:
                    feature_set[v[0]] = (feature_set[v[0]] + v[1]) / 2
                else:
                    feature_set[v[0]] = v[1]
        except TypeError:
            if not override and feature_name in feature_set:
                feature_set[feature_name] += value
            else:
                feature_set[feature_name] = value

    def extract_features(self, sents, window_size, step_size, para_ind_feat=_PARAIND_FEAT, para_dep_feat=_PARADEP_FEAT):
        """Extract and store features with a sliding window."""
        current_pos = 0
        end_pos = current_pos + window_size
        segments = []
        all_sents_done = False

        print "Extracting features ({} sentences)".format(len(sents))
        while(not all_sents_done):
            if end_pos == len(sents):
                all_sents_done = True

            feature_set = {}
            multi_paras = self.in_multiple_paras(current_pos, end_pos-1)

            print "current window: {}-{}\r".format(current_pos, end_pos),
            if multi_paras:
                for f in para_dep_feat:
                    if multi_paras:
                        # compute data for each para separately and compute average afterwards
                        prev_mp = current_pos
                        for mp in multi_paras:
                            upper_bound = min(end_pos, mp+1)
                            self.add_feature(feature_set, f, getattr(features, f)(sents[prev_mp:upper_bound]), False)
                            prev_mp = mp + 1
                # compute average
                # for key in feature_set:
                #     feature_set[key] /= len(multi_paras)

            # if window just hits a single paragraph, just compute features normally, without computing average between paras
            else:
                for f in para_dep_feat:
                    self.add_feature(feature_set, f, getattr(features, f)(sents[current_pos:end_pos]))

            # paragraph independent features
            for f in para_ind_feat:
                self.add_feature(feature_set, f, getattr(features, f)(sents[current_pos:end_pos]))            
            
            data_segment = (current_pos, end_pos-1, feature_set)
            segments.append(data_segment)
            current_pos += step_size
            end_pos = current_pos + window_size
            end_pos = min(end_pos, len(sents))
        return segments

    def in_multiple_paras(self, start, end):
        """Check if section belongs to multiple paragraphs.

        If so: return list of associated paras.
        If not: return False.
        """
        # print "check section {} - {}".format(start, end)
        para_start = -1
        para_end = -1
        para_bounds = None
        for i, elem in enumerate(self.boundaries):
            # print "{} - {}".format(i, elem)
            if para_start == -1 and elem >= start:
                # found 1st ele > start
                para_start = i
                # print "para_start {}".format(para_start)
            if para_start != -1 and elem >= end:
                # found 1st ele > end
                para_end = i
                # print "para_end {}".format(para_end)
                if para_start == para_end:
                    # section belongs to just one para
                    return False
                else:
                    # return list of associated paras
                    para_bounds = list(itertools.islice(self.boundaries, para_start, para_end+1))
                    
                    # remove all elements from boundarie-deque which are of no use anymore
                    temp_bound = self.boundaries[0]
                    while temp_bound != para_bounds[0]:
                        self.boundaries.popleft()
                        temp_bound = self.boundaries[0]
                    return para_bounds

    def unified_features(self, data):
        """Unify and return the features of all sliding-window-instances."""
        features = set()
        feature_sets = self.generator_data(data)
        
        for fset in feature_sets:
            features = features.union(fset.keys())
        return features

    def feature_matrix_sklearn(self, feature_sets):
        """Build and return feature-matrix."""
        X = self.dictVectorizer.fit_transform(feature_sets)
        # print self.dictVectorizer.get_feature_names()
        return X

    def generator_data(self, data):
        """Return a generator which holds the sliding-window-instances of the data."""
        return (feature_set for x, y, feature_set in [datasegment for datasegment in data])