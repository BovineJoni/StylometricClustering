#!/usr/bin/env python
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

"""StylometricClustering - clusterer module
-----------------------------------------------------------

Use:
  Feature extraction and clustering gets started
  automatically if txt- or xml-file is provided as argument. 

Args:
  arg1 (filepath):
    Direct or indirect path to txt- or xml-file.
    Joins path with test-file-path if full path
    is not given.

Note:
  txt-file (UTF-8 encoded):
    Each paragraph seperated by an empty line (\\n\\n).
    Each sentence in a seperate line (if auto-split-
    sentences is deactivated).
  xml-testfile (UTF-8 encoded):
    See ".\\336_Altruism.xml" as reference (auto-split-
    sentences does not apply for xml-files). 

Config file ".\\config.cfg":
  test-file-folder:
    path to test-file-directory - indirect paths 
  auto-split-sentences[0 or 1]:
    0 to deactivate sentence splitting for txt-files,
    1 to activate it.
-----------------------------------------------------------
"""
from __future__ import division
from lxml import etree
from sklearn.metrics.metrics import UndefinedMetricWarning
import cluster_util
import codecs
import ConfigParser
import data
import data_util as util
import io
import matplotlib.pyplot as plt
import numpy
import os
import sys
import time
import warnings
# print __doc__

class Clusterer(object):
    """Organize clustering of a text file."""
    _WINDOW_SIZE = data._WINDOW_SIZE
    _STEP_SIZE = data._STEP_SIZE

    # Feature set
    paragraph_ind_features = ['word_len_bigrams', 'word_len_trigrams', 'mean_word_len', 'word_len_freq', 'short_words', 'medium_words', 'long_words',
        'mean_sent_len', 'short_sents', 'medium_sents', 'long_sents', 'extra_long_sents',
        'char_bigrams_extended', 'freq_consonants', 'char_trigrams',
        'primary_verbs', 'auxiliary_verbs',
        'punctuation_ratio', 'common_punctuation_ratio',
        'function_word', 'pos_freq', 'pos_bigrams',
        'voc_yule', 'voc_hapax_legomenon', 'voc_dis_legomenon']
    paragraph_dep_features = ['sent_len_bigrams', 'sent_len_trigrams', 'sent_len_4grams']

    _IND_FEATURES = paragraph_ind_features
    _DEP_FEATURES = paragraph_dep_features

    def __init__(self, article_id, filename, xml_filename=None, auto_split_sentences=True, show_knee_point=False, ind_features=_IND_FEATURES, dep_features=_DEP_FEATURES):
        """Create clusterer object.

        Args:
          article_id (str): id of the article
          filename (str): path to the text file to cluster
          xml_filename (str, optional): path to the xml-testfile
          auto_split_sentences (bool, optional): option to split sentences into seperate lines
          show_knee_point (bool, optional): option to show knee point detection figure (currently in GUI-mode not avail.)
          ind_features (list, optional): list of paragraph independent features to extract
          dep_features (list, optional): list of paragraph dependent features to extract

        Note:
          If both filename (txt-file) and xml_filename are given, the resulting output
          will evaluate the clustering against the ground truth. Otherwise only the
          clustering and the calculated believe-score get returned.
        """
        self.article_id = article_id
        self.filename = filename
        self.xml_filename = xml_filename
        self.show_knee_point = show_knee_point

        # check if there is an xml-file with the same name as the txt-file
        if self.xml_filename is None:
            xml_filepath = os.path.splitext(filename)[0] + ".xml"
            if os.path.exists(xml_filepath):
                self.xml_filename = xml_filepath
            # do auto sentence splitting only if there is no ground truth data (xml-file)
            elif auto_split_sentences:
                # tokenize sentences and save back to file
                f = io.open(self.filename, mode='r', encoding='utf-8')
                text = ""
                for line in f:
                    text += line
                f.close()  

                tokenized_text = util.return_tokenized_text(text)
                f = io.open(self.filename, mode='w', encoding='utf-8')
                f.write(tokenized_text)
                f.close()

        self.ind_features = ind_features
        self.dep_features = dep_features
        self.xml_tree = self.read_xml()
        self.authors = self.correct_cluster()
        self.labels = self.correct_labels()
        self.labels_pW = self.labels_per_window()
        self.author_no = None
        if self.xml_tree is not None:
            self.author_no = int(self.xml_tree.getroot().get("authors"))

    def correct_cluster(self):
        """Return a dictionary with authors-ids and their corresponding text line numbers."""
        if self.xml_filename is None:
            return None

        new_dict = {}
        for entry in self.xml_tree.getroot():
            author_id = entry.get("author_id")
            start = int(entry.get("start"))
            end = int(entry.get("end"))
            try:
                new_dict[author_id] += range(start, end)
            except KeyError:
                new_dict[author_id] = range(start, end)
        return new_dict

    def correct_labels(self):
        """Return a list of author labels for the text, sorted by line number."""
        if self.xml_filename is None or self.authors is None:
            return None

        num_sents = sum(map(lambda dv: len(dv), self.authors.values()))
        labels = [None for i in range(num_sents)]

        for key in self.authors:
            for index in self.authors[key]:
                labels[index] = key
        return labels

    def labels_per_window(self):
        """Return labels for sliding-window-instances, calculated out of correct cluster label per sentence."""
        if self.xml_filename is None or self.labels is None:
            return None
        return cluster_util.assign_labels(self.labels, self._WINDOW_SIZE, self._STEP_SIZE)


    def read_xml(self):
        """Return parsed xml structure."""
        if self.xml_filename is None:
            return None
        with codecs.open(self.xml_filename, 'r', 'UTF-8') as xml_file:
            return etree.parse(xml_file)
    
    def extract_features(self):
        """Extract features from given text-file."""
        # start = time.clock()
        print("Clustering {}:".format(self.filename))

        d = data.Data(self.article_id)
        self.filename, self.boundaries, self.X, self.num_sents = d.extract_data(self.filename, self.ind_features, self.dep_features, self.labels, self.labels_pW)
        # elapsed = (time.clock() - start)
        # print("time elapsed for feature extraction {}").format(elapsed)

    def calc_cluster(self, show_dendro=False):
        """Cluster data and return results."""
        self.extract_features()

        ward_label, believe_score = cluster_util.cluster_gap(self.X, max(10,int(self.X.shape[0]/2)), self.show_knee_point)
        assigned_clusters = cluster_util.assign_cluster(ward_label, self.num_sents, self.boundaries, self._WINDOW_SIZE, self._STEP_SIZE)
        
        if self.show_knee_point:
            plt.show()
        
        if self.authors is not None:
            # ground truth is known
            result, _, _ = cluster_util.eval_assignment(self.authors, assigned_clusters)
        

            label_set = list(set(self.labels))
            sent_target_labels = dict((label, label_set.index(label)) for label in label_set)
            targets = map(lambda target: sent_target_labels[target], self.labels)

            print "Correct targets: {}".format(set(targets))
            print "{}\n".format(numpy.array(targets))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=UndefinedMetricWarning)
                X_predicted, X_adjusted, scores = cluster_util.evaluate(ward_label, targets, self.num_sents, self.boundaries,self._WINDOW_SIZE, self._STEP_SIZE, self.X)
                cluster_util.print_evaluation(self.labels, X_predicted, X_adjusted, scores)

            print "believe-score: {}".format(believe_score)     
            return assigned_clusters, result, self.author_no, believe_score, (scores["precision"], scores["recall"], scores["f1"], scores["adjusted_rand"])

        else:
            # ground truth is not known
            print "{} authors found with a believe-score of {}:".format(len(set(assigned_clusters)),believe_score)
            print ward_label
            return assigned_clusters, None, None, believe_score, None



def init_clustering(filepath):
    """Initialize clustering with given input file."""
    ext = os.path.splitext(filepath)[1]
    xml_filepath = None

    if ext == ".xml":
        '''save raw-text of xml-file to a new file and print it'''
        xml_filepath = filepath
        txt_filepath = os.path.splitext(filepath)[0] + ".txt"
        util.create_text_fromXML(xml_filepath, txt_filepath)
        filepath = txt_filepath

    base = os.path.split(filepath)[0]
    article_id = os.path.split(base)[1]
    
    if os.path.splitext(filepath)[1] != ".txt":
        print "Input file has to be a .txt file."
        return

    clusterer = Clusterer(article_id, filepath, xml_filepath, auto_split_sentences, show_knee_point)
    clusters, result, author_no, believe_score, scores = clusterer.calc_cluster()
    

if __name__ == '__main__':
    # get config params
    config = ConfigParser.ConfigParser()
    config.read("config.cfg")
    params = dict(config.items("params"))
    article_dir = params['test_file_dir']
    auto_split_sentences = bool(int(params['auto_split_sentences']))
    show_knee_point = bool(int(params['show_knee_point']))

    # example filepath (gets overwritten if script is executed with a filepath-argument)
    filepath = "336_Altruism.txt"
    filepath = os.path.join(article_dir, filepath)

    # check filepath argument
    user_args = sys.argv[1:]
    if len(user_args) > 0:
        filepath = user_args[0]

    if os.path.exists(filepath):
        # use given filepath as is
        init_clustering(os.path.realpath(filepath))
    else:
        # join given filepath with root directory specified in config.cfg
        joined_with_rootdir = os.path.join(article_dir, filepath)
        if os.path.exists(joined_with_rootdir):
            init_clustering(os.path.realpath(joined_with_rootdir))
        else:
            print "File {} does not exist!\n".format(joined_with_rootdir)
            print __doc__
