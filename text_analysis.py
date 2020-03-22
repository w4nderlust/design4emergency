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
def plot_word_cloud(data_df, wordcloud_path, language):
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
    wordcloud.to_file(wordcloud_path)


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
    count_dict = (zip(words, total_counts))
    count_dict = sorted(
        count_dict, key=lambda x: x[1], reverse=True
    )[0:n_top_words]
    return count_dict


##########################
# plot most frequent words
##########################
def plot_most_frequent_words(count_data, count_vectorizer, num_words,
                             frequent_words_plot_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)

    # helper function
    def plot_words(count_dict, output_path):
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        plt.figure(2, figsize=(15, 15 / 1.6180))
        plt.subplot(title='Most common words')
        sns.barplot(x_pos, counts)
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.savefig(output_path)

    # Visualise the 10 most common words
    count_dict = most_frequent_words(count_data, count_vectorizer, num_words)
    plot_words(count_dict, frequent_words_plot_path)


###################
# learn topic model
###################
def learn_topic_model(count_data, count_vectorizer, num_topics, num_words):
    # LDA is a specific kind of topic model
    from sklearn.decomposition import LatentDirichletAllocation as LDA

    # Create and fit the LDA model
    lda = LDA(n_components=num_topics, n_jobs=-1)
    lda.fit(count_data)
    return lda


###################
# print topics
###################
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


#######################
# visualize topic model
#######################
def visualize_topic_model(lda, count_data, count_vectorizer,
                          num_topics, ldavis_path):
    from pyLDAvis import sklearn as sklearn_lda
    import pickle
    import pyLDAvis

    ldavis_data_path = os.path.join(ldavis_path + str(num_topics))
    lbavis_html_path = ldavis_path + str(num_topics) + '.html'
    # this is a bit time consuming - make the if statement True
    # if you want to execute visualization prep yourself
    ldavis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    with open(ldavis_data_path, 'wb') as f:
        pickle.dump(ldavis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(ldavis_data_path, 'rb') as f:
        ldavis_prepared = pickle.load(f)
        pyLDAvis.save_html(ldavis_prepared, lbavis_html_path)


def text_analysis(
        data_path,
        column,
        language,
        num_topics,
        num_words,
        wordcloud_path,
        frequent_words_plot_path,
        ldavis_path,
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
    plot_word_cloud(data_df, wordcloud_path, language)
    print("Wordcloud saved to:", wordcloud_path)
    print()

    count_vectorizer, count_data = get_vectorizer_and_count_data(data_df,
                                                                 language)

    print("Generating frequent word plot...")
    plot_most_frequent_words(count_data, count_vectorizer, num_words,
                             frequent_words_plot_path)
    print("Frequent word plot saved to:", frequent_words_plot_path)
    print()

    print("Calculating topic model...")
    lda = learn_topic_model(count_data, count_vectorizer, num_topics, num_words)
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, num_words)
    print()

    print("Generating LDA visualization...")
    visualize_topic_model(lda, count_data, count_vectorizer,
                          num_topics, ldavis_path)
    print("LDA visualization saved to:", ldavis_path)


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
        'column',
        type=str,
        help='column to extract from TSV'
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
        '--wordcloud_path',
        type=str,
        help='path to save the wordcloud to',
        default='wordcloud.png'
    )
    parser.add_argument(
        '-fw',
        '--frequent_words_plot_path',
        type=str,
        help='path to save the frequent word plot to',
        default='frequent_words.png'
    )
    parser.add_argument(
        '-lv',
        '--ldavis_path',
        type=str,
        help='path (prefix) to save LDA vis plot files to',
        default='ldavis_prepared_'
    )
    args = parser.parse_args()
    text_analysis(**vars(args))
