"""
# Zachary Kimo Stine
# Last updated 2020-06-26
#
# Description:
# - 2 lda models are specified.
# - 1 comparison corpus is specified.
# - High-level measures are calculated to describe the relationship between the two models relative to the comparison
#  corpus.
# - Topic-level measures are calculated to assess the relationships between the features of each model relative to the
#  comparison corpus.
"""
import os
import topic_modeling
import corpus_processing
import numpy as np
import info_measures as info
import csv


def get_doc_topics_for_corpus(doc_topics_path, model_path, corpus_path, k):

    if os.path.exists(doc_topics_path):
        doc_topics = topic_modeling.load_topic_dists(doc_topics_path)

    else:
        lda_model = topic_modeling.load_model(model_path)
        corpus = corpus_processing.load_corpus(corpus_path)

        doc_topics = topic_modeling.write_and_return_topic_dists_memory_friendly(lda_model,
                                                                                 corpus,
                                                                                 k,
                                                                                 doc_topics_path)
    return doc_topics


def write_pmi_csv(pmi_matrix, m1_name, m1_features, m2_name, m2_features, fpath):
    num_pairs = pmi_matrix.shape[0] * pmi_matrix.shape[1]

    indices = np.unravel_index(np.argsort(-1.0 * pmi_matrix, axis=None), pmi_matrix.shape)
    sorted_indices_1 = indices[0]
    sorted_indices_2 = indices[1]

    assert len(sorted_indices_1) == len(sorted_indices_2) == num_pairs

    with open(fpath, 'w', encoding='utf-8', newline='') as ofile:
        fwriter = csv.writer(ofile)

        fwriter.writerow(['pmi_rank', 'pmi', m1_name + '_feature', m2_name + '_feature'])

        for i in range(num_pairs):
            rank = i + 1
            feature_index_1 = sorted_indices_1[i]
            feature_index_2 = sorted_indices_2[i]
            pmi = pmi_matrix[feature_index_1, feature_index_2]
            fwriter.writerow([rank, pmi, m1_features[feature_index_1], m2_features[feature_index_2]])


if __name__ == '__main__':
    experiment_name = 'exp1'

    # Input for model 1, model 2, and target corpus name.
    model_1_corpus_name = 'buddhism'
    model_1_name = '003_k-60'

    model_2_corpus_name = 'christianity'
    model_2_name = '003_k-60'

    comp_corpus_list = ['buddhism', 'christianity']

    # Directories
    cwd = os.getcwd()  # Directory of this program
    project_dir = os.path.dirname(cwd)  # Main project directory
    data_dir = os.path.join(project_dir, '0_data')  # Data directory
    exp_dir = os.path.join(cwd, experiment_name)
    corpora_dir = os.path.join(exp_dir, '2_corpora')
    lda_dir = os.path.join(exp_dir, '3_lda')
    comparison_dir = corpus_processing.make_dir(os.path.join(exp_dir, '5_model_comparisons'))

    # Create a directory for storing any comparisons made between the two models
    results_dir = corpus_processing.make_dir(os.path.join(comparison_dir,
                                                          model_1_corpus_name + '_' + model_1_name + '-' + model_2_corpus_name + '_' + model_2_name))

    # High-level results to be written after all comparison corpora have been analyzed.
    comparison_dict = {}

    # For each unique combination of model_1, model_2, and comp_corpus, first check and see if doc-topic matrix exists,
    # otherwise, create one.
    for comp_corpus_name in comp_corpus_list:

        # path to the corpus may be needed.
        comp_corpus_path = os.path.join(corpora_dir, comp_corpus_name, 'corpus.mm')

        # If model_1 already has doc-topic matrix available, load it, otherwise create it.
        model_1_dir = os.path.join(lda_dir, model_1_corpus_name, model_1_name)
        doc_topics_path_1 = os.path.join(model_1_dir, 'topic_dists', comp_corpus_name + '_tdists.txt')
        lda_model_1_path = os.path.join(model_1_dir, 'model_files', 'lda_model')
        k_1 = int(model_1_name.split('-')[-1])
        doc_topics_1 = get_doc_topics_for_corpus(doc_topics_path_1,
                                                 lda_model_1_path,
                                                 comp_corpus_path,
                                                 k_1)

        # Do likewise for model_2.
        model_2_dir = os.path.join(lda_dir, model_2_corpus_name, model_2_name)
        doc_topics_path_2 = os.path.join(model_2_dir, 'topic_dists', comp_corpus_name + '_tdists.txt')
        lda_model_2_path = os.path.join(model_2_dir, 'model_files', 'lda_model')
        k_2 = int(model_2_name.split('-')[-1])
        doc_topics_2 = get_doc_topics_for_corpus(doc_topics_path_2,
                                                 lda_model_2_path,
                                                 comp_corpus_path,
                                                 k_2)

        # Calculate the mutual information between the two models via the comparison corpus.
        soft_mi, soft_nmi = info.mutual_information(doc_topics_1, doc_topics_2, norm=True)

        # Calculate the entropy of the comparison corpus according to both models.
        entropy_1 = info.entropy(doc_topics_1)
        entropy_2 = info.entropy(doc_topics_2)

        comparison_dict[comp_corpus_name] = {'mi': soft_mi, 'nmi': soft_nmi,
                                             model_1_corpus_name + '_entropy': entropy_1,
                                             model_2_corpus_name + '_entropy': entropy_2}

        soft_pmi = info.pointwise_mutual_info(doc_topics_1, doc_topics_2)
        write_pmi_csv(soft_pmi,
                      model_1_corpus_name, [i for i in range(doc_topics_1.shape[1])],
                      model_2_corpus_name, [i for i in range(doc_topics_2.shape[1])],
                      os.path.join(results_dir, 'pmi_soft-' + comp_corpus_name + '.csv'))

    # write the high-level comparison results to a csv.
    with open(os.path.join(results_dir, 'all_comparisons.csv'), 'w', encoding='utf-8', newline='') as ofile:
        fwriter = csv.writer(ofile)

        fwriter.writerow(['comparison_corpus',
                          'mi',
                          'nmi',
                          model_1_corpus_name + '_entropy',
                          model_2_corpus_name + '_entropy'])

        for comp_corpus_name in comp_corpus_list:
            fwriter.writerow([comp_corpus_name,
                              comparison_dict[comp_corpus_name]['mi'],
                              comparison_dict[comp_corpus_name]['nmi'],
                              comparison_dict[comp_corpus_name][model_1_corpus_name + '_entropy'],
                              comparison_dict[comp_corpus_name][model_2_corpus_name + '_entropy']])
