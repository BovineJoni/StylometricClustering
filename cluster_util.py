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

"""StylometricClustering - clustering util module
-----------------------------------------------------------

Note:
  clustering utilities: cluster assignment and evaluation,
  clustering, knee point detection and believe score 
-----------------------------------------------------------
"""
from __future__ import division
from collections import Counter, OrderedDict
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import numpy as np

"""Cluster assignment and evaluation"""
def span_assignments(flat_clusters, num_sents, window_size, step_size):
    """Span cluster assignment according to window size.

    One sentence has therefore min 1 cluster assigned and max window_size assigned clusters.
    """
    clu_assignments = [[] for i in range(num_sents)]
    step = step_size
    for i, flat_cluster in enumerate(flat_clusters):
        for j in range(window_size):
            try:
                clu_assignments[(i*step)+j].append(flat_cluster)
            except IndexError:
                pass
    return clu_assignments

def correct_assignment(sent_num, counter, assigned_clusters, boundaries, window_size, step_size):
    """Calculates correct cluster for a sentence.

    Calculation influenced by majority of assignments,
    current paragraph and last paragraph assignments.
    """
    most_common = counter.most_common()
    sum_elements = sum(counter.values())
    
    if len(most_common) == 1 or most_common[0][1] >= round((sum_elements / 100) * 66):
        # one cluster has 2/3 majority
        return most_common[0][0]
    
    else:
        # more than one cluster with the same max frequency
        # check clusters of previous and current paragraph
        for i, boundary in enumerate(list(boundaries)):
            # print "sent_num: {}, boundary: {}".format(sent_num, boundary)
            if boundary >= sent_num:
                # found belonging paragraph upper-boundary
                # print "upper boundary: {}".format(boundary)
                lower_boundary = None
                if i != 0:
                    lower_boundary = boundaries[i-1] + 1

                if lower_boundary == sent_num:
                    # sentence is 1st sentence of a new paragraph (= not the 1st paragraph)
                    if i < 2:
                        lower_boundary = 0
                    else:
                        lower_boundary = boundaries[i-2] + 1
                    upper_boundary = boundaries[i-1] + 1
                    most_common_lastpara = Counter(assigned_clusters[lower_boundary:upper_boundary]).most_common(1)[0][0]

                    for mc in most_common:
                        if mc[0] != most_common_lastpara:
                            return mc[0]

                else:
                    # sentence is part of a paragraph already containing some assigned sentences
                    if lower_boundary is None:
                        lower_boundary = 0
                    most_common_currpara = Counter(assigned_clusters[lower_boundary:sent_num]).most_common(1)[0][0]
                    return most_common_currpara

def assign_cluster(flat_cluster, num_sents, boundaries, window_size, step_size):
    """Assign clusters to sentences, calculated out of flat_cluster assignments per window."""
    spanned_clusters = span_assignments(flat_cluster, num_sents, window_size, step_size)
    # print len(spanned_clusters)
    # print "############spanned_clusters###################"
    # for i, clu in enumerate(spanned_clusters):
    #     print i, clu
    assigned_clusters = [None] * len(spanned_clusters)

    for i, cluster_list in enumerate(spanned_clusters):
        counter = Counter(cluster_list)
        assigned_clusters[i] = correct_assignment(i, counter, assigned_clusters, boundaries, window_size, step_size)
    return assigned_clusters

def assign_labels(correct_assignment, window_size, step_size):
    """Assigns labels to windows, calculated out of correct cluster assignment per sentence."""
    # count correct_assignments per window and take majority
    num_windows = int(math.ceil(len(correct_assignment) / step_size))
    correct_labels = [None for i in range(num_windows)]

    for i, label in enumerate(correct_labels):
        most_common = Counter(correct_assignment[(i*step_size):(i*step_size+window_size)]).most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            # either there is just one assigned label or one label has a majority
            correct_labels[i] = most_common[0][0]
        else:
            # more than one label with same frequency
            if i!=0 and correct_labels[i-1] in set([most_common[0][0], most_common[0][1]]):
                # take same label as taken for last window, if label has a majority
                correct_labels[i] = correct_labels[i-1]
            else:
                # take the first you can get
                correct_labels[i] = most_common[0][0]
    return correct_labels

def eval_assignment(real_targets_dict, assigned_targets):
    """Return boolean-list of assigned target-labels evaluated against correct target-labels."""
    mostCommClust_perAuth = {}
    
    # get most common clusters per author indices
    for key in real_targets_dict:
        temp_list = []
        for i in real_targets_dict[key]:
            temp_list.append(assigned_targets[i])
        mostCommClust_perAuth[key] = Counter(temp_list)

    cluster_to_author = {}
    for key in real_targets_dict:
        cluster_to_author[key] = -1
    
    author_to_cluster = {}
    for ele in set(assigned_targets):
        author_to_cluster[ele] = -1

    cluster_set = set(assigned_targets)
    while len(cluster_set) > 0:
        author = None
        mc_cluster = 0
        mc_count = 0
        
        for key, value in mostCommClust_perAuth.iteritems():
            if value.most_common()[0][1] > mc_count:
                mc_count = value.most_common()[0][1]
                mc_cluster = value.most_common()[0][0]
                author = key

        if author is None:
            # more clusters found than actual existing authors
            break

        cluster_to_author[author] = mc_cluster
        author_to_cluster[mc_cluster] = author

        del mostCommClust_perAuth[author]
        empty_authors = []
        for key, value in mostCommClust_perAuth.iteritems():
            del value[mc_cluster]
            if len(value) == 0:
                empty_authors.append(key)
        for ea in empty_authors:
            del mostCommClust_perAuth[ea]

        cluster_set.remove(mc_cluster)

    # map cluster-ids to true target-ids
    ass_clusters_correct = [None for i in range(len(assigned_targets))]
    for key, value in real_targets_dict.iteritems():
        cluster_key = cluster_to_author[key]
        for i in value:
            ass_clusters_correct[i] = cluster_key

    # map target-ids to cluster-ids
    target_to_clusterIDs = [None for i in range(len(assigned_targets))]
    for index, target in enumerate(assigned_targets):
        target_to_clusterIDs[index] = author_to_cluster[target]

    
    result = [None for i in range(len(assigned_targets))]
    for i in range(len(assigned_targets)):
        if assigned_targets[i] == ass_clusters_correct[i]:
            result[i] = True
        else:
            result[i] = False

    # c = Counter(result)
    # correct = c[True] / sum(c.values()) * 100
    # print "true: {} %".format(correct)
    return result, target_to_clusterIDs, cluster_to_author

def evaluate(label, targets, num_sents, boundaries, window_size, step_size, X=None):
    """Return assigned targets, adjusted_targets and evaluation scores."""
    assigned_labels = assign_cluster(label, num_sents, boundaries, window_size, step_size)

    # generate dict of targets {target_id: [target-indices,]}
    targets_dict = dict((target_label, []) for target_label in set(targets))

    for index, target in enumerate(targets):
        targets_dict[target].append(index)
    
    for key, value in targets_dict.iteritems():
        value = sorted(value)

    # compute % of correct assigned
    result, adjusted_assigned_labels, cluster_to_author = eval_assignment(targets_dict, assigned_labels)
    # print cluster_to_author
    # c = Counter(result)
    # correct = c[True] / sum(c.values()) * 100

    scores = OrderedDict()

    if X is not None and len(set(label)) > 1:
        silhouette_scores = ['euclidean', 'manhattan', 'l1', 'l2', 'cityblock', 'cosine', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        silhouette_scores = ['euclidean', 'cosine', 'correlation']

        for sil in silhouette_scores:
            scores["silh_"+sil] = metrics.silhouette_score(X, label, metric=sil)

    scores["adjusted_mutual_info"] = metrics.adjusted_mutual_info_score(targets, assigned_labels)
    scores["adjusted_rand"] = metrics.adjusted_rand_score(targets, assigned_labels)
    scores["homogeneity"] = metrics.homogeneity_score(targets, assigned_labels)
    scores["completeness"] = metrics.completeness_score(targets, assigned_labels)
    scores["v_measure"] = metrics.v_measure_score(targets, assigned_labels)

    try:
        scores["precision"] = metrics.precision_score(targets, adjusted_assigned_labels)
        scores["recall"] = metrics.recall_score(targets, adjusted_assigned_labels)
        scores["f1"] = 2 * ((scores["precision"]*scores["recall"])/(scores["precision"]+scores["recall"]))#        metrics.f1_score(targets, adjusted_assigned_labels)
    except:
        print "Cannot evaluate precision/recall/f1. Number of correct authors is probably 1."
        scores["precision"] = 0.0
        scores["recall"] = 0.0
        scores["f1"] = 0.0
   
    return np.array(assigned_labels), np.array(adjusted_assigned_labels), scores

def print_evaluation(targets_true, X_predicted, X_adjusted, scores):
    """Print evaluation scores."""
    print "{} clusters found: {}".format(len(set(X_predicted)),set(X_predicted))
    print "{}\n".format(X_predicted)
    print "Adjusted labels:\n{}\n".format(X_adjusted)

    # print "mean-score: {}".format(mean_scores(scores))
    for score, value in scores.iteritems():
        print "{} {:.4f}".format(score, float(value))


"""Clustering"""
def cluster_gap(X, num_max_clusters=15, show_knee_point=False):
    """Cluster data and return labels and believe-score.

    Cluster data with Wards agglomerative clustering
    and within cluster variance as relative measure
    to the number of clusters.
    """
    num_max_clusters = max(10, num_max_clusters)
    np.random.seed(1)
    
    varWard = []
    for num_cluster in range(1, num_max_clusters+1):
        ward = AgglomerativeClustering(n_clusters=num_cluster).fit(X)
        label = ward.labels_
        varWard.append(np.mean( [np.var(X[np.where(label == i)], axis=0) for i in set(label)]))      
    
    # knee point detection
    est_clusters, believe_score = normalized_score(varWard, num_max_clusters, gradient="dec", show_knee_point=show_knee_point)

    ward = AgglomerativeClustering(n_clusters=est_clusters).fit(X)
    ward_label = ward.labels_

    return ward_label, believe_score


"""Knee point detection and believe score"""
def normalized_score(t, n_max, n_min=1, gradient="dec", show_knee_point=False):
    """Return number of clusters and believe score.

    Knee point detection based on Zhao et al. (2008),
    'Knee Point Detection on Bayesian Information Criterion'
    20th IEEE International Conference on Tools with
    Artificial Intelligence, 431-438.
    """
    n_max = len(t)

    t_min = min(t)
    t_max = max(t)
    C1 = [ (t[i]-t_min) / (t_max-t_min) for i in range(n_max)]
    Cm = [C1[i] / (pow(i+2,0.95)) for i in range(n_max)]

    if gradient == "dec":
        Cdiff = (np.array(C1) - np.array(Cm)) / 2
    elif gradient == "inc":
        Cdiff = (np.array(C1) + np.array(Cm)) / 2

    est_clusters = np.where(Cdiff==max(Cdiff))[0][0] + 1
    
    # calculate believe score
    Cdiff_2 = np.diff(Cdiff[est_clusters-1:],2)
    consistency = 1 - (sum(local_maxima(Cdiff_2)) - sum(local_minima(Cdiff_2)))
    believe_score = consistency * weight_of_abs_maximum(Cdiff)
    
    if show_knee_point:
        # show knee point detection
        plt.figure(1)
        plt.subplot(111)
        plt.plot(range(1,n_max+1),C1,'r')#r-+
        plt.plot(range(1,n_max+1),Cdiff,'b-+')#b-o
        
        plt.xlabel("Clusteranzahl k")
        plt.ylabel("normalisierte mittlere Varianz")
        plt.show(block=False)

    return est_clusters, believe_score

def weight_of_abs_maximum(t):
    """Return weight of absolute maximum in relation to remaining maxima."""
    max_index = np.where(t==max(t))[0][0]
    abs_maximum_weight = None

    a = np.append(t, t[-1])
    a = np.r_[a[:-1] > a[1:], True] & np.r_[True, a[1:] > a[:-1]]
    a = a[:-1]
    a_ = [False for i in range(len(a))]
    for index, is_min in enumerate(a):
        if is_min:
            end_local_max, end_range = get_end_localmax(t, index)
            start_local_max, start_range = get_start_localmax(t, index)
            r_diff = (t[index] - end_local_max) * end_range
            l_diff = (t[index] - start_local_max) * start_range
            
            # print "l_diff {} - r_diff {}".format(l_diff, r_diff)
            a_[index] = (r_diff + l_diff) / 2 * t[index], min(end_range, start_range)
            if index == max_index:
                abs_maximum_weight = a_[index][0]

    weights = map(lambda filtered_maximum: filtered_maximum[0] ,filter(lambda maximum: maximum is not False and maximum[1] >= a_[max_index][1] , a_))
    return abs_maximum_weight / sum(weights)

def local_minima(t):
    """Return local minima."""
    # a = np.convolve(t, [1,t[-1]])
    a = np.append(t, t[-1])
    a = np.r_[a[:-1] < a[1:], True] & np.r_[True, a[1:] < a[:-1]]
    a = a[:-1]

    a_ = [False for i in range(len(a))]
    for index, is_min in enumerate(a):
        if is_min:
            a_[index] = t[index]
    return a_

def local_maxima(t):
    """Return local maxima."""
    # a = np.convolve(t, [1,t[-1]])
    a = np.append(t, t[-1])
    a = np.r_[a[:-1] > a[1:], True] & np.r_[True, a[1:] > a[:-1]]
    a = a[:-1]

    a_ = [False for i in range(len(a))]
    for index, is_max in enumerate(a):
        if is_max:
            a_[index] = t[index]
    return a_

def get_start_localmax(t, index):
    """Return point where given local maximum starts (bounded to an intervall)."""
    if index == 0:
        return 0, 1

    current_start = 1
    last_grad = 1
    last_value = t[index]
    for index_, i in enumerate(t[index-1::-1]):
        if index_ >= 4:
            break
        current_grad = last_value - i
        if i < current_start:# and current_grad < last_grad:
            current_start = i
            last_grad = current_grad
            last_value = i
        else:
            index_ -= 1
            break
    return current_start, index_+1

def get_end_localmax(t, index):
    """Return point where given local maximum ends (bounded to an intervall)."""
    if index == len(t)-1:
        return 0, 1

    current_end = 1
    last_grad = 1
    last_value = t[index]
    for index_, i in enumerate(t[index+1:]):
        if index_ >= 4:
            break
        current_grad = last_value - i
        if i < current_end:# and current_grad < last_grad:
            current_end = i
            last_grad = current_grad
            last_value = i
        else:
            index_ -= 1
            break
    return current_end, index_+1
