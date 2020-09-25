"""
# Zachary K. Stine | zkstine@ualr.edu
# Last updated 2020-06-23
@
# This script takes the raw data for r/Buddhism, r/Christianity, r/religion, and r/math, and  creates a gensim
# dictionary object that can be used across each subreddit, and creates gensim bag-of-word objects for each subreddit.
"""

import os
import itertools
import corpus_processing


if __name__ == '__main__':
    experiment_name = 'exp1'

    subreddit_list = ['buddhism',
                      'christianity',
                      'math',
                      'religion']

    # Directories
    cwd = os.getcwd()  # Directory of this program
    project_dir = os.path.dirname(cwd)  # Main project directory
    data_dir = os.path.join(project_dir, '0_data')  # Data directory

    # Create directory for storing experiment-specific results
    exp_dir = corpus_processing.make_dir(os.path.join(cwd, experiment_name))

    # Create directory for storing each bag-of-words object
    corpora_dir = corpus_processing.make_dir(os.path.join(exp_dir, '2_corpora'))

    # Read in pre-made set of stopwords
    stoplist = corpus_processing.load_stoplist('stoplist.txt')

    # Open list of bot users to ignore (optional)
    ignore_users = list(set(open('ignore_users.txt', 'r', encoding='utf-8').read().lower().split(',\n')))

    # For each subreddit, create a subreddit-specific dictionary object with the following:
    #   i. Documents included must have at least 20 post-processing tokens or they are ignored.
    #  ii. Tokens must appear within at least five different documents or else they are dropped from the dictionary.
    # iii. Only the 30k most frequent tokens (post-processing) will be kept.

    # Keep each dictionary in the following list:
    dict_list = []

    for subreddit in subreddit_list:

        # Make a list of each document path for the subreddit. Function takes a list of subreddits, hence the [].
        doc_paths = corpus_processing.get_document_paths([subreddit], data_dir)

        # Create gensim dictionary object for subreddit
        subr_dict = corpus_processing.make_dictionary_memory_friendly(doc_paths, stoplist, ignore_users,
                                                                      min_tokens=20, n_below=5, n_keep=30000)

        # Append the subreddit-specific dictionary to the dict_list for later use.
        dict_list.append(subr_dict)

        print('subreddit: ' + subreddit + ' has ' + str(len(subr_dict)) + ' unique postprocessing word types.')

    # Merge the subreddit-specific dictionaries into a single dictionary to allow easier combinations of corpora.
    merged_dictionary = corpus_processing.make_universal_dictionary(dict_list)

    # Save the universal dictionary within the experiment directory.
    dictionary_path = os.path.join(exp_dir, 'dictionary.dict')
    corpus_processing.write_dictionary(dictionary_path, merged_dictionary)

    # Using merged_dictionary as the ID<->token map for each subreddit, create serialized bag-of-word objects for each
    # subreddit-specific corpus.
    for subreddit in subreddit_list:

        # Again, get a list of each document path for the subreddit.
        doc_paths = corpus_processing.get_document_paths([subreddit], data_dir)

        # Use the memory-friendly class to create the bag-of-words corpus object.
        subr_corpus = corpus_processing.MyCorpus(merged_dictionary, doc_paths, stoplist, ignore_users)

        # Make a subreddit-specific directory in corpora_dir.
        subr_corpus_dir = os.path.join(corpora_dir, subreddit)
        corpus_processing.create_dir(subr_corpus_dir)

        # Save the corpus object in the new directory.
        subr_corpus_path = os.path.join(subr_corpus_dir, 'corpus.mm')
        corpus_processing.write_corpus(subr_corpus_path, subr_corpus)

        # Save document-level information that corresponds with the corpus object.
        subr_corpus_data_path = os.path.join(subr_corpus_dir, 'corpus_data.csv')
        subr_corpus.write_corpus_data(subr_corpus_data_path)
