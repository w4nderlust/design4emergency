import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import json
from tqdm import tqdm

STOPWORDS_FILE = 'all_stopwords.json'
with open(STOPWORDS_FILE, encoding="utf8") as json_data:
    stopwords = json.load(json_data)

PARALLEL_DOTS_KEY = ''


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
    data_df = data_df.map(lambda x: re.sub('[,.!?]', '', x))
    data_df = data_df.map(lambda x: re.sub('\s+', ' ', x))
    # Convert the titles to lowercase
    data_df = data_df.map(lambda x: x.strip().lower())

    return data_df


###########
# lemmatize
###########
def lemmatize_text(data_df, language):
    import spacy

    language_map = {
        'it': 'it_core_news_sm',
        'en': 'it_core_web_sm'
    }

    nlp = spacy.load(language_map[language])

    def text2lemmas(text):
        return ' '.join(token.lemma_ for token in nlp(text))

    data_df = data_df.map(text2lemmas)
    return data_df


#################
# plot word cloud
#################
def plot_word_cloud(data_df, word_cloud_filename, language):
    # Import the word cloud library
    from wordcloud import WordCloud

    # plot one word cloud per column
    long_string = ','.join(list(data_df.values))
    # Create a Word Cloud object
    word_cloud = WordCloud(
        background_color="white",
        max_words=5000,
        contour_width=3,
        contour_color='steelblue',
        stopwords=stopwords[language]
    )
    # Generate a word cloud
    word_cloud.generate(long_string)
    # Visualize the word cloud
    word_cloud.to_file(word_cloud_filename)


#####################################
# get count vectorizer and count data
#####################################
def get_count_vectorizer_and_transformed_data(data_df, language, ngram_range):
    from sklearn.feature_extraction.text import CountVectorizer
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words=stopwords[language],
                                       ngram_range=ngram_range)
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(data_df)
    return count_vectorizer, count_data


#####################################
# get tfidf vectorizer and count data
#####################################
def get_tfidf_vectorizer_and_transformed_data(data_df, language, ngram_range):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = TfidfVectorizer(stop_words=stopwords[language],
                                       ngram_range=ngram_range)
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
def save_words(word_count_pair_list, frequent_words_filename):
    word_count_dict = {w: int(c) for w, c in word_count_pair_list}
    with open(frequent_words_filename, "w", encoding="utf8") as f:
        json.dump(word_count_dict, f, ensure_ascii=False)


##########################
# plot most frequent words
##########################
def plot_top_words(word_count_pair_list, frequent_words_plot_filename):
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
        json.dump(topics, f, ensure_ascii=False)


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


###############################################
# predict sentiment with paralleldots (english)
###############################################
def predict_sentiment_with_paralleldots(data_df):
    import paralleldots
    # Setting your API key
    paralleldots.set_api_key(PARALLEL_DOTS_KEY)
    texts_list = data_df.tolist()
    result = paralleldots.sentiment(texts_list)
    return result['sentiment']


###############################################
# predict sentiment with alberto (italian)
###############################################
def predict_sentiment_with_alberto(data_df):
    import requests
    positive_url = "http://193.204.187.35:50000/api/alberto_pos_tw"
    negative_url = "http://193.204.187.35:50000/api/alberto_neg_tw"

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    predicted_sentiment = {'positive': [], 'negative': []}

    texts = data_df.tolist()

    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size)):
        text_batch = texts[i:i + batch_size]
        data = {'messages': text_batch}

        positive_response = requests.post(positive_url, data=json.dumps(data),
                                          headers=headers)
        positive_response = positive_response.json()
        for elem in positive_response['results']:
            predicted_sentiment['positive'].append(elem['class'])

        negative_response = requests.post(negative_url, data=json.dumps(data),
                                          headers=headers)
        negative_response = negative_response.json()
        for elem in negative_response['results']:
            predicted_sentiment['negative'].append(elem['class'])

    return predicted_sentiment


###############################################
# predict sentiment with sentita (italian)
###############################################
def predict_sentiment_with_sentita(data_df):
    from sentita import calculate_polarity
    predicted_sentiment = {'positive': [], 'negative': []}
    texts = data_df.tolist()
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size)):
        text_batch = texts[i:i + batch_size]
        polarities = calculate_polarity(text_batch)
        for elem in polarities:
            predicted_sentiment['positive'].append(elem[0])
            predicted_sentiment['negative'].append(elem[1])
    return predicted_sentiment

################
# save sentiment
################
def save_sentiment(predicted_sentiment, predicted_sentiment_filename):
    predicted_sentiment_df = pd.DataFrame(
        data=predicted_sentiment
    )
    predicted_sentiment_df.to_csv(predicted_sentiment_filename, index=False)


def text_analysis(
        data_path,
        column,
        language,
        lemmatize,
        ngram_range,
        num_topics,
        num_words,
        word_cloud_filename,
        frequent_words_filename,
        frequent_words_plot_filename,
        top_tfidf_words_filename,
        top_tfidf_words_plot_filename,
        topics_filename,
        predicted_topics_filename,
        ldavis_filename_prefix,
        predict_sentiment,
        predicted_sentiment_filename,
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

    if lemmatize:
        print("Lemmatizing data...")
        data_df = lemmatize_text(data_df, language)
        print("Lemmatized data sample")
        print(data_df.head())
        print()

    print("Generating word cloud...")
    plot_word_cloud(data_df, word_cloud_filename, language)
    print("word_cloud saved to:", word_cloud_filename)
    print()

    count_vectorizer, count_data = get_count_vectorizer_and_transformed_data(
        data_df, language, ngram_range
    )
    word_count_pair_list = most_frequent_words(
        count_data, count_vectorizer, num_words
    )
    tfidf_vectorizer, tfidf_data = get_tfidf_vectorizer_and_transformed_data(
        data_df, language, ngram_range
    )

    print("Saving frequent words...")
    save_words(
        most_frequent_words(count_data, count_vectorizer, count_data.shape[0]),
        frequent_words_filename
    )
    print("Frequent words saved to:", frequent_words_filename)
    print()

    print("Generating frequent word plot...")
    plot_top_words(word_count_pair_list, frequent_words_plot_filename)
    print("Frequent word plot saved to:", frequent_words_plot_filename)
    print()

    print("Saving top tfidf words...")
    save_words(
        most_frequent_words(tfidf_data, tfidf_vectorizer, tfidf_data.shape[0]),
        top_tfidf_words_filename
    )
    print("Top tfidf words saved to:", top_tfidf_words_filename)
    print()

    print("Generating frequent word plot...")
    plot_top_words(word_count_pair_list, top_tfidf_words_plot_filename)
    print("Frequent word plot saved to:", top_tfidf_words_plot_filename)
    print()

    print("Calculating topic model...")
    lda, predicted_topics = learn_topic_model(tfidf_data, num_topics)
    print("Topics found via LDA:")
    print_topics(lda, tfidf_vectorizer, num_words)
    print("Saving topics...")
    save_topics(lda, tfidf_vectorizer, topics_filename)
    print("Topics saved to:", topics_filename)
    print()

    print("Saving predicted topics...")
    save_predicted_topics(predicted_topics, predicted_topics_filename)
    print("Predicted topics saved to:", predicted_topics_filename)
    print()

    print("Generating LDA visualization...")
    visualize_topic_model(lda, count_data, tfidf_vectorizer,
                          num_topics, ldavis_filename_prefix)
    print("LDA visualization saved to:", ldavis_filename_prefix)
    print()

    if predict_sentiment:
        if language == 'it':
            print("Predict sentiment...")
            predicted_sentiment = predict_sentiment_with_sentita(data_df)
            save_sentiment(predicted_sentiment, predicted_sentiment_filename)
            print("Predict sentiment saved to:", predicted_sentiment_filename)
            print()
        elif language == 'en':
            print("Predict sentiment...")
            predicted_sentiment = predict_sentiment_with_paralleldots(data_df)
            save_sentiment(predicted_sentiment, predicted_sentiment_filename)
            print("Predict sentiment saved to:", predicted_sentiment_filename)
            print()
        else:
            print("Sentiment analysis on {} language is not supported")
            print()


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
        description='This script analyzes text in columns of a TSV file'
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
        '-lm',
        '--lemmatize',
        action='store_true',
        help='performs lemmatization of all texts',
    )
    parser.add_argument(
        '-nr',
        '--ngram_range',
        type=str,
        help='minimum and maximum value for ngrams, specify as "min,max"',
        default="1,1"
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
        '--word_cloud_filename',
        type=str,
        help='path to save the word cloud to',
        default='word_cloud.png'
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
        '-ttw',
        '--top_tfidf_words_filename',
        type=str,
        help='path to save top tfidf words to',
        default='top_tfidf_words.json'
    )
    parser.add_argument(
        '-ttwp',
        '--top_tfidf_words_plot_filename',
        type=str,
        help='path to save the top tfidf word plot to',
        default='top_tfidf_words.png'
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
        '-s',
        '--predict_sentiment',
        action='store_true',
        help='performs sentiment analysis (it is pretty slow)',
    )
    parser.add_argument(
        '-ps',
        '--predicted_sentiment_filename',
        type=str,
        help='path to save predicted sentiment for each datapoint to',
        default='predicted_sentiment.csv'
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        help='path that will contain all directories, one for each column',
        default='.'
    )
    args = parser.parse_args()

    ngram_range = (1, 1)
    try:
        ngram_range = tuple([int(s) for s in args.ngram_range.split(',')])
    except:
        print('ngram_range is not properly formatted: {}. '
              'Please use the format "min,max"'.format(args.ngram_range))
        exit(-1)

    for column in args.columns:
        column_dir = format_filename(column)
        if not os.path.exists(column_dir):
            os.makedirs(column_dir)

        word_cloud_filename = os.path.join(
            args.output_path, column_dir, args.word_cloud_filename
        )
        frequent_words_filename = os.path.join(
            args.output_path, column_dir, args.frequent_words_filename
        )
        frequent_words_plot_filename = os.path.join(
            args.output_path, column_dir, args.frequent_words_plot_filename
        )
        top_tfidf_words_filename = os.path.join(
            args.output_path, column_dir, args.top_tfidf_words_filename
        )
        top_tfidf_words_plot_filename = os.path.join(
            args.output_path, column_dir, args.top_tfidf_words_plot_filename
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
        predicted_sentiment_filename = os.path.join(
            args.output_path, column_dir, args.predicted_sentiment_filename
        )

        text_analysis(
            data_path=args.data_path,
            column=column,
            language=args.language,
            lemmatize=args.lemmatize,
            ngram_range=ngram_range,
            num_topics=args.num_topics,
            num_words=args.num_words,
            word_cloud_filename=word_cloud_filename,
            frequent_words_filename=frequent_words_filename,
            frequent_words_plot_filename=frequent_words_plot_filename,
            top_tfidf_words_filename=top_tfidf_words_filename,
            top_tfidf_words_plot_filename=top_tfidf_words_plot_filename,
            topics_filename=topics_filename,
            predicted_topics_filename=predicted_topics_filename,
            ldavis_filename_prefix=ldavis_filename_prefix,
            predict_sentiment=args.predict_sentiment,
            predicted_sentiment_filename=predicted_sentiment_filename
        )
