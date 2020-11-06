"""
# Zachary Kimo Stine
# 2019-09-17, updated 2020-11-06
#
# This file contains all of the functions needed to convert collected data into a gensim-friendly format along with some
# basic preprocessing.
"""

import os
import ast
import html
import re
from nltk.probability import FreqDist
import csv
import string
from urllib.parse import urlparse
import math
import numpy as np
import gensim


class MyCorpus(object):

    def __init__(self, dict, corp_paths, stoplist, ignore_users):
        self.dict = dict
        self.corp_paths = corp_paths
        self.stoplist = stoplist
        self.ignore_users = [u.lower() for u in ignore_users]

    def __iter__(self):
        for doc_path in self.corp_paths:
            doc_data = get_doc_tokens_from_path(doc_path, self.stoplist, self.ignore_users, min_tokens=20)
            if doc_data is not None:
                doc = doc_data[0]
                yield self.dict.doc2bow(doc)

    def write_corpus_data(self, path):

        with open(path, 'w', newline='') as ofile:
            fwriter = csv.writer(ofile)
            fwriter.writerow(['corpus_index', 'submission_id', 'submission_date', 'submission_score', 'doc_length',
                              'submission_url', 'subreddit'])

            doc_index = 0

            for doc_path in self.corp_paths:
                doc_data = get_doc_tokens_from_path(doc_path, self.stoplist, self.ignore_users, min_tokens=20)

                if doc_data is not None:
                    doc, id, date, score = doc_data

                    # Get the subreddit directory name from the document_path
                    subreddit = get_subreddit_from_submission_path(doc_path)

                    # Construct a URL to the submission from the subreddit name and the submission ID.
                    submission_url = 'www.reddit.com/r/' + subreddit + '/comments/' + id + '/'

                    fwriter.writerow([doc_index, id, date, score, len(doc), submission_url, subreddit])

                    doc_index += 1


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def write_word_freqs_and_props(fpath, dictionary, corpus):
    """
    :param dictionary: Gensim dictionary object that maps each word type to index (or ID).
    :param corpus: Gensim corpus object.
    :return: vectors of word frequencies and relative frequencies from corpus.
    """
    word_counts = np.zeros(len(dictionary))

    for bow_doc in corpus:
        for (word_id, count) in bow_doc:
            word_counts[word_id] += count

    total_tokens = np.sum(word_counts)
    word_proportions = np.divide(word_counts, total_tokens, dtype=np.float64)

    with open(fpath, 'w', newline='', encoding='utf-8') as ofile:
        fwriter = csv.writer(ofile)
        fwriter.writerow(['word_id', 'word', 'word_count', 'word_proportion'])

        for word_id in dictionary:
            fwriter.writerow([word_id, dictionary[word_id], word_counts[word_id], word_proportions[word_id]])

    return word_counts, word_proportions


def load_word_freqs_and_props(fpath, dictionary):
    """
    :param fpath: File path where word frequencies are written.
    :param dictionary: Gensim dictionary object that specifies which vector index each word type is assigned to.
    :return: Vectors of word frequencies and relative frequencies.
    """
    word_counts = np.zeros(len(dictionary))
    word_props = np.zeros(len(dictionary))

    with open(fpath, 'r', encoding='utf-8') as infile:
        freader = csv.reader(infile)
        header = True

        for row in freader:
            if header:
                header = False
                continue

            wid = int(row[0])
            wcount = float(row[2])
            wprop = float(row[3])

            word_counts[wid] = wcount
            word_props[wid] = wprop

    return word_counts, word_props


def get_subreddit_from_submission_path(subm_path):

    # Get the subreddit directory name from the document_path
    sub_dir = os.path.dirname(os.path.dirname(subm_path))

    # The basename of the subreddit directory is 'r_subreddit'. Only take 'subreddit' hence the [2:].
    subreddit = os.path.basename(sub_dir)[2:]

    return subreddit


def convert_url_to_domain(matchobject):
    """
    This function is used by convert_text_to_tokens() to reformat urls.
    """
    parsed = urlparse(matchobject.group(0))
    return parsed.scheme + '-' + parsed.netloc.replace('.', '-')


def convert_text_to_tokens(comment_text, stoplist):
    """
    :param comment_text: String of comment text.
    :param stoplist: Set of tokens to remove.
    :return: A list of tokens.

    This is a reddit-specific process that takes into account urls, subreddit names, and redditor names.
    """

    comment_string = comment_text.replace('[deleted]', ' ').replace('[removed]', ' ')

    # For urls, subreddit names, and redditor names, use only dashes as in-token separators:

    # 1) Convert all urls to their domain-names and get rid of surrounding parentheses and brackets.
    # Ex: 'https://www.en.wikipedia.org/ -> https-en-wikipedia-org
    url_pattern = re.compile(r'http\S+')
    procd_urls_string = re.sub(url_pattern,
                               convert_url_to_domain,
                               comment_string.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' '))

    # Now create two lists of tokens: one from the string without the urls and the other only having the urls
    no_urls_tokens = [t for t in procd_urls_string.split(' ') if t[:5] != 'http-' and t[:6] != 'https-']
    urls_tokens = [t for t in procd_urls_string.split(' ') if t[:5] == 'http-' or t[:6] == 'https-']

    # 2) Similarly, for all subreddit names, replace slashes with dashes. Start from no_urls_tokens.
    # Ex: '/r/buddhism' -> 'r-buddhism' AND 'r/buddhism' -> 'r-buddhism'
    no_subreddits_tokens = [t for t in no_urls_tokens if t[:3] != '/r/' and t[:2] != 'r/']
    subreddits_tokens = ['r-' + t[3:] for t in no_urls_tokens if t[:3] == '/r/'] + \
                        ['r-' + t[2:] for t in no_urls_tokens if t[:2] == 'r/']

    # 3) Same thing for redditor names. Ex: '/u/redditor' -> 'u-redditor' and 'u/redditor' -> 'u-redditor'
    no_redditors_tokens = [t for t in no_subreddits_tokens if t[:3] != '/u/' and t[:2] != 'u/']
    redditors_tokens = ['u-' + t[3:] for t in no_subreddits_tokens if t[:3] == '/u/'] + \
                       ['u-' + t[2:] for t in no_subreddits_tokens if t[:2] == 'u/']

    # 4) It's now safe to remove all punctuation, including dashes, from no_redditors_tokens
    punc_pattern = r'[{}]'.format(string.punctuation)
    no_punc_string = re.sub(punc_pattern, ' ', ' '.join(no_redditors_tokens))

    # 5) Remove numeric characters and convert string back to tokens.
    no_nums_string = re.sub(r'[\d\W_]', ' ', no_punc_string)
    no_nums_tokens = no_nums_string.split(' ')

    # 6) With the cleaned tokens in hand, add the urls, subreddits, and redditors back in. Remove tokens with length
    # less than 3.
    combined_tokens = no_nums_tokens + urls_tokens + subreddits_tokens + redditors_tokens
    cleaned_tokens = [t for t in combined_tokens if len(t) > 2 and t not in stoplist]

    return cleaned_tokens


def get_doc_tokens_from_path(doc_path, stoplist, ignore_users, min_tokens=20):
    """
    :param doc_path: File path for a document.
    :param stoplist: Stopwords to be removed.
    :param min_tokens: Minimum number of tokens the document must have to be included.
    :return: If document has the minimum # of tokens or more, a list of tokens and other data from the submission:
    ID, date, & score.
    """
    ignore_users = [u.lower() for u in ignore_users]

    submission_tokens = []

    submission_filename = os.path.basename(doc_path)
    submission_date = submission_filename.split('_')[0]
    submission_id = submission_filename.split('_')[1][:-4]

    with open(doc_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        for comment in reader:
            # The first row of each CSV corresponds to the original submission post, indicated by the 'type' column.
            if comment['type'] == 'submission':
                submission_score = int(comment['score'])

            # For each comment, do not include those generated by bots.
            if comment['author_name'].lower() not in ignore_users:

                # Read in comment text as byte string by evaluating and decode from utf-8
                comment_text_raw = ast.literal_eval(comment['text']).decode('utf-8').lower()

                # Convert hexadecimal html codes into unicode characters (e.g. '&#8212;' becomes an em-dash)
                comment_text = html.unescape(comment_text_raw)

                # If this the boilerplate "i left reddit for voat" message, then move on to the next comment
                if comment_text[:29] == 'i have left reddit for [voat]':
                    continue

                # Convert comment text into tokens and add to submission tokens
                comment_tokens = convert_text_to_tokens(comment_text, stoplist)

                # submission_tokens.append(comment_tokens)
                submission_tokens += comment_tokens

        # Add all submission data to the relevant lists if it meets the min_tokens threshold.
    if len(submission_tokens) >= min_tokens:
        return submission_tokens, submission_id, submission_date, submission_score
    else:
        return None


def get_most_frequent_tokens(sub_path_list, min_tokens=1):
    """
    :param sub_path_list: List of paths for each subreddit's submission directory.
    :return: Frequency distribution of all word types from any subreddits in sub_path_lst.
    """

    # Initialize the frequency distribution object that we will iteratively add to.
    fd = FreqDist()

    for sub_dir in sub_path_list:

        for submission in os.listdir(sub_dir):

            submission_path = os.path.join(sub_dir, submission)

            # no stoplist is used when looking at frequent tokens, so an empty list is passed instead.
            submission_data = get_doc_tokens_from_path(submission_path, [], [], min_tokens=min_tokens)

            if submission_data is not None:
                processed_tokens = submission_data[0]

                # add the token frequencies from the current submission to the global frequency distribution object.
                fd += FreqDist(processed_tokens)

    return fd


def make_universal_dictionary(dictionary_list):
    """
    :param dictionary_list: List of Gensim dictionary objects.
    :return: Gensim dictionary object that combines the vocabularies of each dictionary object in dictionary_list.
    """
    # Use the first dictionary object in dictionary_list as the dictionary that all others will merge into.
    merged_dictionary = dictionary_list[0]

    # Iterate through the remaining dictionaries and merge each into merged_dictionary.
    for other_dictionary in dictionary_list[1:]:

        # No need to save the transformer since the universal dictionary will be made prior to creating the BoW docs.
        transformer = merged_dictionary.merge_with(other_dictionary)

    return merged_dictionary


def make_dictionary_memory_friendly(doc_path_list, stoplist, ignore_users, min_tokens=20, n_below=5, n_above=100.0, n_keep=100000):
    """
    :param doc_path_list: List of file paths for all documents to be used in making the dictionary object.
    :param stoplist: List of stopwords to be removed.
    :param ignore_users: List of users whose comments will not be included. This is used for bot accounts that create a lot of repeated text.
    :param min_tokens: Minimum number of tokens a document must have to be included.
    :param n_below: If a word type occurs in fewer than n_below documents, it is not included.
    :param n_above: If a word type occurs in more than n_above percent of documents, it is not included.
    :param n_keep: After other words have been filtered, keep the n_keep most frequent terms and discard the rest.
    :return: Gensim dictionary object that maps each word type to an index or ID value.
    """
    dictionary = gensim.corpora.Dictionary()

    for doc_path in doc_path_list:
        doc_data = get_doc_tokens_from_path(doc_path, stoplist, ignore_users, min_tokens=min_tokens)

        if doc_data is not None:
            doc_tokens = doc_data[0]
            dictionary.add_documents([doc_tokens])

    dictionary.filter_extremes(no_below=n_below, no_above=n_above, keep_n=n_keep)

    return dictionary


def get_document_paths(sub_list, data_dir, min_date=None):
    """
    :param sub_list: list of subreddits that document paths will be obtained for.
    :param data_dir: directory where subreddit data is stored.
    :return: list of paths to each document file in CSV format.
    """
    doc_paths = []

    for subreddit in sub_list:
        submissions_dir = os.path.join(data_dir, 'r_' + subreddit, 'threads')

        for submission_file in os.listdir(submissions_dir):

            if min_date is not None:
                submission_date = int(submission_file.split('_')[0])

                if submission_date >= min_date:
                    submission_path = os.path.join(submissions_dir, submission_file)
                    doc_paths.append(submission_path)

            else:
                submission_path = os.path.join(submissions_dir, submission_file)
                doc_paths.append(submission_path)

    return doc_paths


def write_dictionary(dict_path, dictionary):
    dictionary.save(dict_path)


def load_dictionary(dict_path):
    """
    :param dict_path: Path to Gensim dictionary object.
    :return: Dictionary object.
    """
    dict = gensim.corpora.Dictionary.load(dict_path)
    return dict


def load_corpus(corpus_path):
    """
    :param corpus_path: Path to serialized Gensim corpus object.
    :return: Corpus object.
    """
    corpus = gensim.corpora.MmCorpus(corpus_path)
    return corpus


def write_corpus(corpus_path, corpus):
    gensim.corpora.MmCorpus.serialize(corpus_path, corpus)


def write_corpus_stats(stats_path, num_docs, num_unique_terms, num_words):
    with open(stats_path, 'w') as ofile:
        ofile.write(str({'num_docs': num_docs, 'num_vocab': num_unique_terms, 'num_tokens': num_words}))


def load_stoplist(fpath):
    """
    :param fpath: File containing stopwords
    :return: Set of stopwords.
    """
    return set(open(fpath, 'r').read().split(',\n'))


def read_corpus_data(fpath):
    """
    :param fpath: Path to a CSV file containing the corresponding data for a corpus object.
    :return: List of dictionaries where each dictionary gives information about the specific document.
    """
    data_list = []
    with open(fpath, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            data_list.append(row)
    return data_list
