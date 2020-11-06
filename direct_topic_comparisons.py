"""
# Zachary Kimo Stine | zkstine@ualr.edu
# 2020-07-14, updated: 2020-11-06
#
# Directly compare topics between two LDA models using the Jensen-Shannon divergence between each pair of topics.
"""

import os
import topic_modeling
import corpus_processing
import csv


def write_pairwise_topic_jsd(fpath, topic_jsds, name_1, name_2):
    with open(fpath, 'w', encoding='utf-8', newline='') as ofile:
        fwriter = csv.writer(ofile)

        fwriter.writerow([name_1 + '_topic_index', name_2 + '_topic_index', 'jsd'])

        for t_index_1 in range(topic_jsds.shape[0]):
            for t_index_2 in range(topic_jsds.shape[1]):
                fwriter.writerow([t_index_1, t_index_2, topic_jsds[t_index_1, t_index_2]])


if __name__ == '__main__':
    experiment_name = 'exp1'

    # Specify the two models to be compared.
    model_1_corpus_name = 'buddhism'
    model_1_name = '001_k-30'

    model_2_corpus_name = 'buddhism'
    model_2_name = '002_k-30'

    # Directories
    cwd = os.getcwd()  # Directory of this program
    project_dir = os.path.dirname(cwd)  # Main project directory
    data_dir = os.path.join(project_dir, '0_data')  # Data directory
    exp_dir = os.path.join(cwd, experiment_name)
    corpora_dir = os.path.join(exp_dir, '2_corpora')
    lda_dir = os.path.join(exp_dir, '3_lda')
    comparison_dir = corpus_processing.make_dir(os.path.join(exp_dir, '9_topic_comparisons'))

    # Create a directory for storing any comparisons made between the two models
    results_dir = corpus_processing.make_dir(os.path.join(comparison_dir,
                                                          model_1_corpus_name + '_' + model_1_name + '-' + model_2_corpus_name + '_' + model_2_name))

    # load both models
    lda_model_1 = topic_modeling.load_model(os.path.join(lda_dir,
                                                         model_1_corpus_name,
                                                         model_1_name,
                                                         'model_files', 'lda_model'))

    lda_model_2 = topic_modeling.load_model(os.path.join(lda_dir,
                                                         model_2_corpus_name,
                                                         model_2_name,
                                                         'model_files', 'lda_model'))

    # create difference matrix where rows represent model_1 topics and columns represent model_2 topics.
    jsd_matrix, annotations = lda_model_1.diff(lda_model_2, distance='jensen_shannon')
    write_pairwise_topic_jsd(os.path.join(results_dir, 'all_pairwise_jsd.csv'),
                             jsd_matrix,
                             model_1_corpus_name,
                             model_2_corpus_name)
