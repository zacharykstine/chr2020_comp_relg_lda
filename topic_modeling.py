"""
# Zachary Kimo Stine | zktine@ualr.edu
# 2020-03-16, updated 2020-11-06
#
# Functions for working with LDA models.
"""

import gensim
import os
import csv
import numpy as np
import logging
import corpus_processing


class ShuffledCorpus(object):
    """
    Corpus object that is yielded based on the order of document indices specified by shuffled_indices.
    """

    def __init__(self, original_corpus, shuffled_indices):
        self.original_corpus = original_corpus
        self.shuffled_indices = shuffled_indices

    def __iter__(self):
        for random_index in self.shuffled_indices:
            yield self.original_corpus[random_index]


def get_shuffled_corpus_indices(corpus):
    """
    Randomly shuffles document indices from a corpus.
    """
    num_docs = len(corpus)
    corpus_indices = [i for i in range(num_docs)]
    np.random.shuffle(corpus_indices)
    return corpus_indices


def train_lda_model(corpus, dictionary, k, random_state, log_path=None, passes=10, iterations=400, eval_every=100):
    """
    Trains LDA model via Gensim.
    """

    if log_path is not None:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO,
                            handlers=[logging.FileHandler(log_path, 'w', 'utf-8')])

    # Train the model
    lda_model = gensim.models.LdaModel(corpus, num_topics=k, id2word=dictionary, passes=passes, iterations=iterations,
                                       eval_every=eval_every, minimum_probability=0.000000001,
                                       alpha='auto', random_state=random_state)

    return lda_model


def write_model_description(fpath, info_dict):
    """
    :param fpath: File path where model description is written.
    :param info_dict: Dictionary containing some basic info about the LDA model that is written to file.
    """
    with open(fpath, 'w') as ofile:
        ofile.write(str(info_dict))


def write_top_words(model, model_dir, k, num_words=500):

    for topic_index in range(k):

        # Make directory to store all topic-specific data
        topic_dir = corpus_processing.make_dir(os.path.join(model_dir, 'topic_' + f'{topic_index:02d}'))

        # Get the highest probability terms from this topic
        topic_words = model.show_topic(topic_index, num_words)

        # Path to CSV file that will hold the topic words and probabilities:
        wordlist_fname = os.path.join(topic_dir, 'topic_' + f'{topic_index:02d}' + '_word_list.csv')

        # Write top terms to csv file.
        with open(wordlist_fname, 'w', newline='', encoding='utf-8') as ofile:
            f_writer = csv.writer(ofile)

            f_writer.writerow(['word', 'probability'])

            for (word, prob) in topic_words:
                f_writer.writerow([word, prob])


def write_exemplary_docs(doc_topics, model_dir, k, corpus_data, num_docs=50):

    # For each topic, find which documents that topic has the highest proportion in.
    for topic_index in range(k):

        # Make sure directory for topic exists
        topic_dir = corpus_processing.make_dir(os.path.join(model_dir, 'topic_' + f'{topic_index:02d}'))

        # Sort document indices from corpus from largest to lowest topic probabilities. doc_topics[:, topic_index]
        # specifies the topic column from the document-topic matrix. This column is sorted from largest to smallest and
        # the row/document indices are returned in this order.
        sorted_doc_indices = np.argsort(-doc_topics[:, topic_index])[:num_docs]
        sorted_topic_probs = [doc_topics[doc_indx, topic_index] for doc_indx in sorted_doc_indices]

        # Specify the path for the topic submission file.
        topic_doclist_path = os.path.join(topic_dir, 'topic_' + f'{topic_index:02d}' + '_document_list.csv')

        # Write exemplary documents for each topic to file.
        with open(topic_doclist_path, 'w', newline='', encoding='utf-8') as ofile:
            fwriter = csv.writer(ofile)

            fwriter.writerow(['submission_rank', 'subreddit', 'topic_probability',
                              'submission_date', 'submission_id', 'submission_url'])

            # For each exemplary document's index, write some useful info about the document to a CSV row.
            for i, doc_index in enumerate(sorted_doc_indices):

                # Exemplar document rank for the submission.
                subm_rank = i + 1

                # Submisison info for this exemplary document.
                doc_data = corpus_data[doc_index]

                # Write information for exemplary document to CSV row
                fwriter.writerow([subm_rank, doc_data['subreddit'], sorted_topic_probs[i],
                                  doc_data['submission_date'], doc_data['submission_id'], doc_data['submission_url']])


def write_topic_summary_file(fpath, model, k, doc_topics, num_words=20):

    # Number of documents in the document-topic matrix
    total_docs = np.shape(doc_topics)[0]

    # Average proportion of each topic across all documents in the document-topic matrix.
    avg_topic_proportions = np.mean(doc_topics, axis=0, dtype=np.float64)

    with open(fpath, 'w', newline='', encoding='utf-8') as ofile:
        fwriter = csv.writer(ofile)
        fwriter.writerow(['topic_index', 'n_docs_dominant', 'p_docs_dominant', 'avg_proportion'] +
                         ['word_' + str(i + 1) for i in range(num_words)])

        for topic_index in range(k):

            # Highest probability words as tuples of form (word, probability).
            topic_words = model.show_topic(topic_index, num_words)

            # Number of documents where this topic is dominant.
            n_docs_dominant = sum([1 for tdist in doc_topics if np.argsort(-tdist)[0] == topic_index])

            # Proportion of total docs where this topic is dominant.
            p_docs_dominant = float(n_docs_dominant) / float(total_docs)

            # Write information to CSV row.
            fwriter.writerow([topic_index, n_docs_dominant, p_docs_dominant, avg_topic_proportions[topic_index]] +
                             [str(w) for w in topic_words])


def load_model(model_path):
    model = gensim.models.LdaModel.load(model_path)
    return model


def write_and_return_topic_dists_memory_friendly(model, corpus, k, path):

    doc_num = 0
    with open(path, 'w') as ofile:
        for doc in corpus:

            doc_dist = model[doc]

            doc_distribution = np.zeros(len(doc_dist), dtype='float64')

            for (topic, val) in doc_dist:
                doc_distribution[topic] = val

            np.savetxt(ofile, doc_distribution.reshape(1, k))
            doc_num += 1

    doc_topics = np.loadtxt(path)
    return doc_topics


def load_topic_dists(fpath):
    """
    :param fpath: Path to a txt file containing a document-topic matrix.
    :return: A numpy matrix containing the specified document-topic matrix.
    There was presumably a reason for making this its own function.
    """
    return np.loadtxt(fpath)
