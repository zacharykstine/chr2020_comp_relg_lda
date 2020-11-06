"""
# Zachary Kimo Stine | zkstine@ualr.edu
# 2019-02-20, 2020-11-06
#
# A collection of functions for calculating relevant information theoretic quantities.
"""

import numpy as np
import math


def entropy(x):
    """
    Given a probability distribution, returns Shannon entropy. Any elements of zero are ignored.

    If x has 2 dimensions, it is assumed to be a document-topic matrix. The column means represent the average
    proportion of each topic across all documents. This is based on the SI appendix for Grimmer & King (2011) A General
    Purpose Computer-Assisted Clustering Methodology, but it's worth remembering that Shannon entropy is not defined
    for a collection of probability distributions, so this is really just the Shannon entropy of the collection's
    mean distribution.

    This way of calculating the entropy of a document-topic matrix also works out with how Lei et al. (2014) define
    their calculation of mutual information between probabilistic clusterings. See mutual_information() and lei_mi().
    """

    if len(x.shape) == 2:
        avg_topic_props = np.mean(x, axis=0)
        x = avg_topic_props

    # Make sure the distribution sums to something near 1.
    assert np.around(np.sum(x), decimals=6) == 1.0, 'entropy(x): x does not sum to 1.'

    # Calculate Shannon entropy, assuming 0log(0) == 0 and so can be skipped.
    h = 0.0
    for i in np.nditer(x):
        if i > 0.0:
            h += i * math.log(i, 2)
    return -h


def joint_entropy(x, y):
    """
    Joint entropy between two document-topic matrices, based on the approach given in Lei et al. (2014). See
    mutual_information() and lei_mi() below for more details.
    """
    contingency_counts = np.matmul(x.T, y)
    contingency_probs = contingency_counts / x.shape[0]
    joint_h = entropy(contingency_probs.flatten())

    return joint_h


def per_features_joint_entropy(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    contingency_counts = np.matmul(x.T, y)
    contingency_probs = contingency_counts / x.shape[0]
    joint_entropy_matrix = np.zeros(contingency_probs.shape)

    for x_i in range(x.shape[1]):
        for y_j in range(y.shape[1]):
            p_xy = contingency_probs[x_i, y_j]
            if p_xy > 0.0:
                joint_entropy_matrix[x_i, y_j] = -1.0 * p_xy * math.log(p_xy, 2)

    return joint_entropy_matrix


def conditional_entropy(x, y):
    """
    :param p:
    :param q:
    :return:

    H(X | Y) = H(X, Y) - H(Y)
    """
    h_xy = joint_entropy(x, y)
    h_y = entropy(y)
    return h_xy - h_y


def mutual_information(x, y, norm=False):
    """
    For probabilistic clusterings given by LDA, mi is calculated based on Lei et al. (2014):
    https://doi.org/10.1109/CIDM.2014.7008144

    Here, we use the following definition of mi:
    I(X; Y) = H(X) + H(Y) - H(X,Y)

    where H(X)

    """
    h_x = entropy(x)
    h_y = entropy(y)
    h_xy = joint_entropy(x, y)

    mi = h_x + h_y - h_xy

    if norm:
        return mi, ((2 * mi) / (h_x + h_y))
    else:
        return mi


def lei_mi(x, y):
    """
    Based on https://doi.org/10.1109/CIDM.2014.7008144 (Lei et al, 2014).

    Provided as a check to make sure the same result is obtained for document-topic matrices as the mutual_information()
    function above. It does with some extremely small differences at very high precision (10 decimals or more).
    """
    # Form a "contingency table" between the two clusterings provided by x and y.
    cont_table = np.matmul(x.T, y)

    n = np.sum(cont_table)
    x_sums = np.sum(cont_table, axis=1)
    y_sums = np.sum(cont_table, axis=0)

    mi = 0.0
    for i in range(cont_table.shape[0]):
        for j in range(cont_table.shape[1]):
            ij = cont_table[i, j]

            pjoint_ij = np.divide(ij, n, dtype=np.float64)

            pmargn_ij = np.divide(x_sums[i] * y_sums[j], math.pow(n, 2), dtype=np.float64)

            mi += pjoint_ij * math.log(pjoint_ij / pmargn_ij, 2)

    return mi


def pointwise_mutual_info(x, y):
    """
    This calculation of pmi is also based on the approach of Lei et al. (2014). See mutual_information() and lei_mi()
    above for more details.
    """

    assert x.shape[0] == y.shape[0]

    p_x = np.mean(x, axis=0, dtype=np.float64)
    p_y = np.mean(y, axis=0, dtype=np.float64)

    assert np.around(np.sum(p_x), decimals=6) == 1.0, 'P(x) does not sum to 1.'
    assert np.around(np.sum(p_y), decimals=6) == 1.0, 'P(y) does not sum to 1. '

    xy_counts = np.matmul(x.T, y)
    xy_probs = xy_counts / x.shape[0]

    pmi_matrix = np.zeros(xy_probs.shape)

    for xi in range(xy_probs.shape[0]):
        for yj in range(xy_probs.shape[1]):
            p_xy = xy_probs[xi, yj]
            pxi = p_x[xi]
            pyj = p_y[yj]

            if p_xy > 0:
                pmi_xy = math.log(p_xy / (pxi * pyj), 2)

            else:
                pmi_xy = 0.0

            pmi_matrix[xi, yj] = pmi_xy

    return pmi_matrix


def js_divergence(x, y):
    """
    Jensen-Shannon divergence calculated as H(M) - (H(X) + H(Y)) / 2, which is equivalent to the definition based on
    the Kullback-Leibler divergence: 0.5*KLD(X|M) + 0.5*KLD(Y|M).
    """
    assert len(x) == len(y), 'js_divergence(): x and y do not have the same number of elements.'
    m = np.mean([x, y], axis=0, dtype=np.float64)
    h_m = entropy(m)
    h_x = entropy(x)
    h_y = entropy(y)
    return h_m - ((h_x + h_y) / 2.0)


def partial_jsds(x, y, proportions=False):
    """
    Jensen-Shannon divergence contribution of each element in distributions x and y. Summing these for all elements
    is equal to js_divergence(x, y) above with some small differences at very high precisions.
    """
    assert len(x) == len(y)

    m = np.mean([x, y], axis=0, dtype=np.float64)

    jsd_list = []

    for i in range(len(x)):
        if m[i] != 0.0:
            hm = -1.0 * m[i] * math.log(m[i], 2)
        else:
            hm = 0.0

        if x[i] != 0.0:
            hx = -1.0 * x[i] * math.log(x[i], 2)
        else:
            hx = 0.0

        if y[i] != 0.0:
            hy = -1.0 * y[i] * math.log(y[i], 2)
        else:
            hy = 0.0

        jsd_list.append(hm - ((hx + hy) / 2.0))

    jsd_array = np.array(jsd_list)

    if proportions:
        jsd_props = np.divide(jsd_array, np.sum(jsd_array), dtype=np.float64)
        return jsd_array, jsd_props

    else:
        return jsd_array


def kl_divergence(x, y):
    """
    Kullback-Leibler divergence from y to x. Or the entropy of x relative to y.
    """
    assert len(x) == len(y)

    kld = 0.0
    for i in range(len(x)):
        if x[i] != 0.0:
            kld += x[i] * math.log(x[i] / y[i], 2)
    return kld


def partial_klds(x, y):
    """
    Individual contributions of each element to the Kullback-Leibler divergence of y to x. This is not the same thing
    as the partial KL calculation used in Klingenstein, Hitchcock, & DeDeo (2014) which is given by the
    partial_klds_to_m() function given below.
    """
    assert len(x) == len(y)
    kl_list = []

    for i in range(len(x)):
        if x[i] == 0.0:
            kl_list.append(0.0)

        elif x[i] > 0.0 and y[i] == 0.0:
            kl_list.append(float('nan'))

        else:
            kl_list.append(x[i] * math.log(x[i] / y[i], 2))

    return np.array(kl_list)


def partial_klds_to_m(x, y):
    """
    Contributions of each element to the Kullback-Leibler divergence, KLD(x, m) where m is the mean of x and y. I.e.,
    this gives the individual contributions from y to x in one side of the Jensen-Shannon divergence between x and y.
    This is what Klingenstein, Hitchcock, & DeDeo (2014) call the partial KL. See https://doi.org/10.1073/pnas.1405984111.
    """
    assert len(x) == len(y)

    m = np.mean([x, y], axis=0, dtype=np.float64)
    partial_kl_list = []

    for i in range(len(x)):
        if x[i] == 0.0:
            partial_kl_list.append(0.0)

        else:
            pkl_i = x[i] * math.log(x[i] / m[i], 2)
            partial_kl_list.append(pkl_i)

    return np.array(partial_kl_list)
