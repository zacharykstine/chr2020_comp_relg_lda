"""
# Zachary K. Stine | zkstine@ualr.edu
# Last updated: 2020-09-22
#
# Description:
# For pairs of subreddits being compared, calculates several word-level information measures that can be used to order
# words by how much they contribute to the differences between the subreddits.
"""

import os
import itertools
import corpus_processing
import info_measures as info


def write_vocabulary_distribution_comparison(dictionary,
                                             corp1_name, corp1_counts, corp1_props,
                                             corp2_name, corp2_counts, corp2_props,
                                             fpath):

    per_word_jsds, jsd_props = info.partial_jsds(corp1_props, corp2_props, proportions=True)
    per_word_klds_1_2 = info.partial_klds(corp1_props, corp2_props)
    per_word_klds_2_1 = info.partial_klds(corp2_props, corp1_props)
    per_word_kld_m_1_2 = info.partial_klds_to_m(corp1_props, corp2_props)
    per_word_kld_m_2_1 = info.partial_klds_to_m(corp2_props, corp1_props)

    with open(fpath, 'w', newline='', encoding='utf=8') as ofile:
        fwriter = csv.writer(ofile)
        fwriter.writerow(['word_id', 'word',
                          'word_jsd', 'jsd_proportion',
                          'word_count_' + corp1_name, 'word_proportion_' + corp1_name,
                          'word_count_' + corp2_name, 'word_proportion_' + corp2_name,
                          'word_kld_' + corp1_name + '-' + corp2_name,
                          'word_kld_' + corp2_name + '-' + corp1_name,
                          'word_kld_to_mean_' + corp1_name + '-' + corp2_name,
                          'word_kld_to_mean_' + corp2_name + '-' + corp1_name])

        for w_id in dictionary:
            fwriter.writerow([w_id, dictionary[w_id],
                              per_word_jsds[w_id], jsd_props[w_id],
                              corp1_counts[w_id], corp1_props[w_id],
                              corp2_counts[w_id], corp2_props[w_id],
                              per_word_klds_1_2[w_id],
                              per_word_klds_2_1[w_id],
                              per_word_kld_m_1_2[w_id],
                              per_word_kld_m_2_1[w_id]])


if __name__ == '__main__':
    experiment_name = 'exp1'

    comparisons = [('buddhism', 'christianity'),
                   ('buddhism', 'math'),
                   ('buddhism', 'religion'),
                   ('christianity', 'math'),
                   ('christianity', 'religion')]

    # Directories
    cwd = os.getcwd()  # Directory of this program
    project_dir = os.path.dirname(cwd)  # Main project directory
    data_dir = os.path.join(project_dir, '0_data')  # Data directory
    exp_dir = os.path.join(cwd, experiment_name)
    corpora_dir = os.path.join(exp_dir, '2_corpora')

    comparison_dir = corpus_processing.make_dir(os.path.join(exp_dir, '4_comparisons'))

    dictionary_path = os.path.join(exp_dir, 'dictionary.dict')
    dictionary = corpus_processing.load_dictionary(dictionary_path)

    for (subreddit_1, subreddit_2) in comparisons:
        sub_comp_dir = corpus_processing.make_dir(os.path.join(comparison_dir, subreddit_1 + '-' + subreddit_2))

        word_freqs_path_1 = os.path.join(corpora_dir, subreddit_1, 'word_freqs.csv')
        word_freqs_path_2 = os.path.join(corpora_dir, subreddit_2, 'word_freqs.csv')

        if os.path.exists(word_freqs_path_1):
            word_counts_1, word_props_1 = corpus_processing.load_word_freqs_and_props(word_freqs_path_1, dictionary)
        else:
            corpus_1_path = os.path.join(corpora_dir, subreddit_1, 'corpus.mm')
            corpus_1 = corpus_processing.load_corpus(corpus_1_path)
            word_counts_1, word_props_1 = corpus_processing.write_word_freqs_and_props(word_freqs_path_1, dictionary, corpus_1)

        if os.path.exists(word_freqs_path_2):
            word_counts_2, word_props_2 = corpus_processing.load_word_freqs_and_props(word_freqs_path_2, dictionary)
        else:
            corpus_2_path = os.path.join(corpora_dir, subreddit_2, 'corpus.mm')
            corpus_2 = corpus_processing.load_corpus(corpus_2_path)
            word_counts_2, word_props_2 = corpus_processing.write_word_freqs_and_props(word_freqs_path_2, dictionary, corpus_2)

        write_vocabulary_distribution_comparison(dictionary,
                                                 subreddit_1, word_counts_1, word_props_1,
                                                 subreddit_2, word_counts_2, word_props_2,
                                                 os.path.join(sub_comp_dir, 'vocab_comparison.csv'))