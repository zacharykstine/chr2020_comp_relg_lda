"""
#
#
#
"""
import os
import numpy as np
import corpus_processing
import topic_modeling

if __name__ == '__main__':

    experiment_name = 'exp1'

    # Designate model ids, k values, and random ints for each round of models to be trained.
    model_ids = ['001', '002', '003', '004', '005', '006']
    k_list = [30, 30, 60, 60, 90, 90]
    random_states = np.random.choice(1000, 6, replace=False)

    assert len(model_ids) == len(k_list) == len(random_states)

    # Fixed modeling parameters
    passes = 20
    eval_every = None
    iterations = 400

    subreddit_list = ['buddhism', 'christianity', 'math', 'religion']

    # Directories
    cwd = os.getcwd()  # Directory of this program
    project_dir = os.path.dirname(cwd)  # Main project directory
    exp_dir = os.path.join(cwd, experiment_name)
    corpora_dir = os.path.join(exp_dir, '2_corpora')

    lda_dir = corpus_processing.make_dir(os.path.join(exp_dir, '3_lda'))

    # Dictionary is universal for each subreddit in this experiment, so only need to read it in once.
    dictionary_path = os.path.join(exp_dir, 'dictionary.dict')
    dictionary = corpus_processing.load_dictionary(dictionary_path)

    # Train model for each subreddit and write all relevant information to files.
    for subreddit in subreddit_list:
        print(subreddit)

        # Load corpus.
        corpus_path = os.path.join(corpora_dir, subreddit, 'corpus.mm')
        corpus = corpus_processing.load_corpus(corpus_path)

        # Load corpus data.
        corpus_data_path = os.path.join(corpora_dir, subreddit, 'corpus_data.csv')
        corpus_data = corpus_processing.read_corpus_data(corpus_data_path)

        # For each set of model specifications, train a model the current subreddit.
        for m_index, model_id in enumerate(model_ids):
            print('  model id: ' + model_id)
            k = k_list[m_index]
            random_state = random_states[m_index]

            # Create model-specific directory
            model_dir = corpus_processing.make_dir(os.path.join(lda_dir, subreddit, model_id + '_k-' + str(k)))

            # Specify file path for logging model progress
            log_path = os.path.join(model_dir, 'model.log')

            # For model training, shuffle the documents in the corpus.
            shuffled_indices = topic_modeling.get_shuffled_corpus_indices(corpus)
            shuffled_corpus = topic_modeling.ShuffledCorpus(corpus, shuffled_indices)

            # Specify a model path.
            model_files_dir = corpus_processing.make_dir(os.path.join(model_dir, 'model_files'))
            model_path = os.path.join(model_files_dir, 'lda_model')

            # Train the model.
            model = topic_modeling.train_lda_model(shuffled_corpus, dictionary, k, random_state,
                                                   log_path=log_path, passes=passes, iterations=iterations,
                                                   eval_every=eval_every)

            # Save model to file.
            model.save(model_path)

            # Write some basic info about the model to file.
            model_description_path = os.path.join(model_dir, 'model_description.txt')

            model_description = {'k': k,
                                 'random_state': random_state,
                                 'passes': passes,
                                 'iterations': iterations,
                                 'eval_every': eval_every}

            topic_modeling.write_model_description(model_description_path, model_description)

            # Write highest probability words to file for reference.
            topic_modeling.write_top_words(model, model_dir, k, num_words=100)

            # Create document-topic matrix using the regular corpus (not the shuffled corpus).
            topic_dist_dir = os.path.join(model_dir, 'topic_dists')
            corpus_processing.create_dir(topic_dist_dir)
            doc_topics_path = os.path.join(topic_dist_dir, subreddit + '_tdists.txt')
            doc_topics = topic_modeling.write_and_return_topic_dists_memory_friendly(model, corpus, k, doc_topics_path)

            # Write topics summary file
            topic_summary_path = os.path.join(model_dir, 'topics_summary.csv')
            topic_modeling.write_topic_summary_file(topic_summary_path, model, k, doc_topics, num_words=20)

            # Write lists of exemplary documents for each topic.
            topic_modeling.write_exemplary_docs(doc_topics, model_dir, k, corpus_data, num_docs=50)

            print('  ' + model_id + ' done.')

        print('__________\n')
