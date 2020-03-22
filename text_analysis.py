import argparse
import os
import pandas as pd
import json

STOPWORDS_FILE = 'all_stopwords.json'
with open(STOPWORDS_FILE, encoding="utf8") as json_data:
    stopwords = json.load(json_data)


###########
# load data
###########
# Read data into dataframe
def load_data(data_path, column):
    data_df = pd.read_csv(data_path, sep='\t', encoding='utf8')
    data_df = data_df.rename(columns={c: c.strip() for c in data_df.columns})
    data_df = data_df[column]
    return data_df


############
# clean data
############
def clean_data(data_df):
    # Load the regular expression library
    import re

    # fill empty rows with empty string
    data_df.fillna('', inplace=True)
    # Remove punctuation
    data_df = data_df.map(lambda x: re.sub('[,\.!?]', '', x))
    data_df = data_df.map(lambda x: re.sub('\s+', ' ', x))
    # Convert the titles to lowercase
    data_df = data_df.map(lambda x: x.strip().lower())

    return data_df


#################
# plot word cloud
#################
def plot_word_cloud(data_df, wordcloud_filename, language):
    # Import the wordcloud library
    from wordcloud import WordCloud

    # plot one wordcloud per column
    long_string = ','.join(list(data_df.values))
    # Create a WordCloud object
    wordcloud = WordCloud(
        background_color="white",
        max_words=5000,
        contour_width=3,
        contour_color='steelblue',
        stopwords=stopwords[language]
    )
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_file(wordcloud_filename)


###############################
# get vectorizer and count data
###############################
def get_vectorizer_and_count_data(data_df, language):
    from sklearn.feature_extraction.text import CountVectorizer
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words=stopwords[language])
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(data_df)
    return count_vectorizer, count_data


#############################
# compute msot frequent words
#############################
def most_frequent_words(count_data, count_vectorizer, n_top_words):
    import numpy as np

    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]
    word_count_pair_list = (zip(words, total_counts))
    word_count_pair_list = sorted(
        word_count_pair_list, key=lambda x: x[1], reverse=True
    )[0:n_top_words]
    return word_count_pair_list


##########################
# save most frequent words
##########################
def save_frequent_words(word_count_pair_list, frequent_words_filename):
    word_count_dict = {w: int(c) for w, c in word_count_pair_list}
    with open(frequent_words_filename, "w", encoding="utf8") as f:
        json.dump(word_count_dict, f)


##########################
# plot most frequent words
##########################
def plot_most_frequent_words(word_count_pair_list, frequent_words_plot_filename):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)

    words = [w[0] for w in word_count_pair_list]
    counts = [w[1] for w in word_count_pair_list]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='Most common words')
    sns.barplot(x_pos, counts, palette='GnBu_r')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.savefig(frequent_words_plot_filename)


###################
# learn topic model
###################
def learn_topic_model(count_data, num_topics):
    # LDA is a specific kind of topic model
    from sklearn.decomposition import LatentDirichletAllocation as LDA

    # Create and fit the LDA model
    lda = LDA(n_components=num_topics, n_jobs=-1)
    predicted_topics = lda.fit_transform(count_data)
    return lda, predicted_topics


##############
# print topics
##############
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


#############
# save topics
#############
def save_topics(model, count_vectorizer, topics_filename):
    topics = {}
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        topics["topic_" + str(topic_idx + 1)] = {
            words[i]: s for i, s in enumerate(topic)
        }
    with open(topics_filename, "w", encoding="utf8") as f:
        json.dump(topics, f)


#####################################
# save predicted topics per datapoint
#####################################
def save_predicted_topics(predicted_topics, predicted_topics_filename):
    predicted_topics_df = pd.DataFrame(
        data=predicted_topics,
        columns=["topic_" + str(i + 1) for i in range(predicted_topics.shape[1])]
    )
    predicted_topics_df.to_csv(predicted_topics_filename, index=False)


#######################
# visualize topic model
#######################
def visualize_topic_model(lda, count_data, count_vectorizer,
                          num_topics, ldavis_filename_prefix):
    from pyLDAvis import sklearn as sklearn_lda
    import pickle
    import pyLDAvis

    ldavis_data_path = os.path.join(ldavis_filename_prefix + str(num_topics))
    ldavis_html_path = ldavis_filename_prefix + str(num_topics) + '.html'
    # this is a bit time consuming - make the if statement True
    # if you want to execute visualization prep yourself
    ldavis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    with open(ldavis_data_path, 'wb') as f:
        pickle.dump(ldavis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(ldavis_data_path, 'rb') as f:
        ldavis_prepared = pickle.load(f)
        pyLDAvis.save_html(ldavis_prepared, ldavis_html_path)


def text_analysis(
        data_path,
        column,
        language,
        num_topics,
        num_words,
        wordcloud_filename,
        frequent_words_filename,
        frequent_words_plot_filename,
        topics_filename,
        predicted_topics_filename,
        ldavis_filename_prefix,
):
    print("Loading data...")
    data_df = load_data(data_path, column)
    print("Loaded data sample")
    print(data_df.head())
    print()

    print("Cleaning data...")
    data_df = clean_data(data_df)
    print("Clean data sample")
    print(data_df.head())
    print()

    print("Generating word cloud...")
    plot_word_cloud(data_df, wordcloud_filename, language)
    print("Wordcloud saved to:", wordcloud_filename)
    print()

    count_vectorizer, count_data = get_vectorizer_and_count_data(data_df,
                                                                 language)
    word_count_pair_list = most_frequent_words(count_data, count_vectorizer,
                                               num_words)

    print("Saving frequent words...")
    save_frequent_words(
        most_frequent_words(count_data, count_vectorizer, count_data.shape[0]),
        frequent_words_filename
    )
    print("Frequent words saved to:", frequent_words_filename)
    print()

    print("Generating frequent word plot...")
    plot_most_frequent_words(word_count_pair_list, frequent_words_plot_filename)
    print("Frequent word plot saved to:", frequent_words_plot_filename)
    print()

    print("Calculating topic model...")
    lda, predicted_topics = learn_topic_model(count_data, num_topics)
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, num_words)
    print("Saving topics...")
    save_topics(lda, count_vectorizer, topics_filename)
    print("Topics saved to:", topics_filename)
    print("Saving predicted topics...")
    save_predicted_topics(predicted_topics, predicted_topics_filename)
    print("Predicted topics saved to:", predicted_topics_filename)
    print()

    print("Generating LDA visualization...")
    visualize_topic_model(lda, count_data, count_vectorizer,
                          num_topics, ldavis_filename_prefix)
    print("LDA visualization saved to:", ldavis_filename_prefix)


def format_filename(s):
    import string
    """Take a string and return a valid filename constructed from the string.
    Uses a whitelist approach: any characters not present in valid_chars are
    removed. Also spaces are replaced with underscores.
    
    Note: this method may produce invalid filenames such as ``, `.` or `..`
    When I use this method I prepend a date string like '2009_01_15_19_46_32_'
    and append a file extension like '.txt', so I avoid the potential of using
    an invalid filename.
    
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')
    filename = filename.lower()
    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script trains a model',
        prog='ludwig train',
        usage='%(prog)s [options]'
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='path to the data TSV'
    )
    parser.add_argument(
        'columns',
        type=str,
        nargs='+',
        help='columns to extract from TSV'
    )
    parser.add_argument(
        '-l',
        '--language',
        type=str,
        help='language of the text in the data (for data cleaning purposes)',
        default='it'
    )
    parser.add_argument(
        '-w',
        '--num_words',
        type=int,
        help='number of most frequent words to show',
        default=10
    )
    parser.add_argument(
        '-t',
        '--num_topics',
        type=int,
        help='number of topics for topic modeling',
        default=5
    )
    parser.add_argument(
        '-wc',
        '--wordcloud_filename',
        type=str,
        help='path to save the wordcloud to',
        default='wordcloud.png'
    )
    parser.add_argument(
        '-fw',
        '--frequent_words_filename',
        type=str,
        help='path to save frequent words to',
        default='frequent_words.json'
    )
    parser.add_argument(
        '-fwp',
        '--frequent_words_plot_filename',
        type=str,
        help='path to save the frequent word plot to',
        default='frequent_words.png'
    )
    parser.add_argument(
        '-tp',
        '--topics_filename',
        type=str,
        help='path to save frequent words to',
        default='topics.json'
    )
    parser.add_argument(
        '-pt',
        '--predicted_topics_filename',
        type=str,
        help='path to save predicted LDA topics for each datapoint to',
        default='predicted_topics.csv'
    )
    parser.add_argument(
        '-lv',
        '--ldavis_filename_prefix',
        type=str,
        help='path (prefix) to save LDA vis plot files to',
        default='ldavis_'
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        help='path that will contain all directories, one for each column',
        default='.'
    )
    args = parser.parse_args()

    for column in args.columns:
        column_dir = format_filename(column)
        if not os.path.exists(column_dir):
            os.makedirs(column_dir)
            
        wordcloud_filename = os.path.join(
            args.output_path, column_dir, args.wordcloud_filename
        )
        frequent_words_filename = os.path.join(
            args.output_path, column_dir, args.frequent_words_filename
        )
        frequent_words_plot_filename = os.path.join(
            args.output_path, column_dir, args.frequent_words_plot_filename
        )
        topics_filename = os.path.join(
            args.output_path, column_dir, args.topics_filename
        )
        predicted_topics_filename = os.path.join(
            args.output_path, column_dir, args.predicted_topics_filename
        )
        ldavis_filename_prefix = os.path.join(
            args.output_path, column_dir, args.ldavis_filename_prefix
        )

        text_analysis(
            data_path=args.data_path,
            column=column,
            language=args.language,
            num_topics=args.num_topics,
            num_words=args.num_words,
            wordcloud_filename=wordcloud_filename,
            frequent_words_filename=frequent_words_filename,
            frequent_words_plot_filename=frequent_words_plot_filename,
            topics_filename=topics_filename,
            predicted_topics_filename=predicted_topics_filename,
            ldavis_filename_prefix=ldavis_filename_prefix
        )
