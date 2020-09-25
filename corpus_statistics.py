"""
# Zachary Kimo Stine | zkstine@ualr.edu
# 2020-03-12, updated 2020-09-14
#
# For a a group of subreddits, want to get some basic stats about submissions collected from them. Most
# importantly, want to know the frequency and relative frequency of each word type within a subreddit's documents.
"""

import os
import csv
import corpus_processing


def write_word_frequenices(fpath, fd, t_count):
    """
    :param fpath:
    :param fd:
    :param t_count:
    :return:
    """
    with open(fpath, 'w', encoding='utf-8', newline='') as ofile:
        fwriter = csv.writer(ofile)
        fwriter.writerow(['word', 'count', 'prop_of_total_tokens'])

        for word, count in fd.most_common(len(fd)):
            prop_of_total_tokens = float(count) / float(t_count)
            fwriter.writerow([word, count, prop_of_total_tokens])


if __name__ == '__main__':
    # Subreddits to be analyzed.
    subreddit_list = ['buddhism', 'christianity', 'religion', 'math']

    # Directories
    cwd = os.getcwd()
    project_dir = os.path.dirname(cwd)
    data_dir = os.path.join(project_dir, '0_data')
    exp_dir = corpus_processing.make_dir(os.path.join(cwd, 'exp1'))
    stats_dir = corpus_processing.make_dir(os.path.join(exp_dir, 'corpus_stats'))

    # Open list of bot users to ignore (optional)
    ignore_users = list(set(open('ignore_users.txt', 'r', encoding='utf-8').read().split(',\n')))

    # Iterate though each subreddit to get some general statistics about its submissions.
    for subreddit in subreddit_list:
        # Directory for subreddit-specific results
        sub_stats_dir = corpus_processing.make_dir(os.path.join(stats_dir, subreddit))

        # Directory where subreddit's submission files are stored.
        submission_dir = os.path.join(data_dir, 'r_' + subreddit, 'threads')

        total_docs = len(os.listdir(submission_dir))
        earliest_submission_day = min([int(os.path.basename(f).split('_')[0]) for f in os.listdir(submission_dir)])
        latest_submission_day = max([int(os.path.basename(f).split('_')[0]) for f in os.listdir(submission_dir)])

        # Get word frequencies.
        freq_dist = corpus_processing.get_most_frequent_tokens([submission_dir])

        vocab_size = len(freq_dist)  # Number of unique word types.
        token_count = sum([freq_dist[w] for w in freq_dist.keys()])  # Number of total tokens.

        # Write words and frequency to csv
        word_freqs_path = os.path.join(sub_stats_dir, 'word_freqs.csv')
        write_word_frequenices(word_freqs_path, freq_dist, token_count)

        # Write some high-level stats to file.
        summary_info = {'earliest_submission_date': earliest_submission_day,
                        'latest_submission_date': latest_submission_day,
                        'total_submissions': total_docs,
                        'vocab_size': vocab_size,
                        'total_tokens': token_count}

        with open(os.path.join(sub_stats_dir, 'subreddit_overview.txt'), 'w') as ofile:
            ofile.write(str(summary_info))
