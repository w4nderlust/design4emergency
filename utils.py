import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

STOPWORDS_FILE = 'it_stopwords.json'
with open(STOPWORDS_FILE, 'r', encoding="utf8") as json_data:
    stopwords = json.load(json_data)
PARALLEL_DOTS_KEY = ''


def load_data(data_path, column, groups=None):
    data_df = pd.read_csv(data_path, sep='\t', encoding='utf8')
    data_df = data_df.rename(columns={c: c.strip() for c in data_df.columns})
    data_df = data_df[[column] + (groups if groups else [])]
    return data_df


def clean_data(data_df):
    # Load the regular expression library
    import re

    # fill empty rows with empty string
    data_df.fillna('', inplace=True)
    # Remove punctuation
    data_df = data_df.map(lambda x: re.sub('[,.!?]', ' ', x))
    data_df = data_df.map(lambda x: re.sub('\s+', ' ', x))
    # Convert the titles to lowercase
    data_df = data_df.map(lambda x: x.strip().lower())

    return data_df


def remove_stopwords(data_df, language):
    data_df = data_df.apply(
        lambda x: ' '.join(
            [word for word in x.split() if word not in stopwords[language]]
        )
    )
    return data_df


def lemmatize_text(data_df, language):
    import spacy

    language_map = {
        'it': 'it_core_news_sm',
        'en': 'it_core_web_sm'
    }

    nlp = spacy.load(language_map[language])

    def text2lemmas(text):
        processed_text = nlp(text)
        lemmas = [token.lemma_ for token in processed_text]
        lemamtized_text = ' '.join(lemmas)
        return lemamtized_text

    data_df = data_df.map(text2lemmas)
    return data_df


def apply_manual_mappings(data_df, manual_mappings):
    with open(manual_mappings, encoding="utf8") as json_data:
        manual_mappings_dict = json.load(json_data)

    def map_text(text):
        return ' '.join(
            manual_mappings_dict[t] if t in manual_mappings_dict else t
            for t in text.split()
        )

    data_df = data_df.map(map_text)
    return data_df


def plot_word_cloud(data_df, word_cloud_filename, language):
    # Import the word cloud library
    from wordcloud import WordCloud

    # plot one word cloud per column
    long_string = ','.join(list(data_df.values))
    # Create a Word Cloud object
    word_cloud = WordCloud(
        background_color="black",
        width=1000,
        height=1000,
        prefer_horizontal=1,
        max_words=200,
        colormap="summer",
        stopwords=stopwords[language]
    )
    # Generate a word cloud
    word_cloud.generate(long_string)
    # Visualize the word cloud
    word_cloud.to_file(word_cloud_filename)


def get_count_vectorizer_and_transformed_data(data_df, language, ngram_range):
    from sklearn.feature_extraction.text import CountVectorizer
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words=stopwords[language],
                                       ngram_range=ngram_range)
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(data_df)
    return count_vectorizer, count_data


def get_tfidf_vectorizer_and_transformed_data(data_df, language, ngram_range):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = TfidfVectorizer(stop_words=stopwords[language],
                                       ngram_range=ngram_range)
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(data_df)
    return count_vectorizer, count_data


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


def save_words(word_count_pair_list, frequent_words_filename):
    word_count_dict = {w: int(c) for w, c in word_count_pair_list}
    with open(frequent_words_filename, "w", encoding="utf8") as f:
        json.dump(word_count_dict, f, ensure_ascii=False)


def plot_top_words(word_count_pair_list, frequent_words_plot_filename):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')

    words = [w[0] for w in word_count_pair_list]
    counts = [w[1] for w in word_count_pair_list]
    y_pos = np.arange(len(words))

    fig, ax = plt.subplots()
    ax.barh(y_pos, counts, align='center')
    ax.set_xlabel('counts')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_title('Most common words')
    plt.tight_layout()
    plt.savefig(frequent_words_plot_filename)


def learn_topic_model(count_data, num_topics):
    # LDA is a specific kind of topic model
    from sklearn.decomposition import LatentDirichletAllocation as LDA

    # Create and fit the LDA model
    lda = LDA(n_components=num_topics, n_jobs=-1)
    predicted_topics = lda.fit_transform(count_data)
    return lda, predicted_topics


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def save_topics(model, count_vectorizer, topics_filename):
    topics = {}
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        topics["topic_" + str(topic_idx + 1)] = {
            words[i]: s for i, s in enumerate(topic)
        }
    with open(topics_filename, "w", encoding="utf8") as f:
        json.dump(topics, f, ensure_ascii=False)


def save_predicted_topics(predicted_topics, predicted_topics_filename):
    predicted_topics_df = pd.DataFrame(
        data=predicted_topics,
        columns=["topic_" + str(i + 1) for i in range(predicted_topics.shape[1])]
    )
    predicted_topics_df.to_csv(predicted_topics_filename, index=False)


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


def predict_sentiment_with_paralleldots(data_df):
    import paralleldots
    # Setting your API key
    paralleldots.set_api_key(PARALLEL_DOTS_KEY)
    texts_list = data_df.tolist()
    result = paralleldots.sentiment(texts_list)
    return result['sentiment']


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


def predict_sentiment_with_sentita(data_df):
    from sentita import calculate_polarity
    predicted_sentiment = {'positive': [], 'negative': []}
    texts = data_df.tolist()
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size)):
        text_batch = texts[i:i + batch_size]
        _, polarities = calculate_polarity(text_batch)
        for elem in polarities:
            predicted_sentiment['positive'].append(elem[0])
            predicted_sentiment['negative'].append(elem[1])
    return predicted_sentiment


def save_sentiment(predicted_sentiment, predicted_sentiment_filename):
    predicted_sentiment_df = pd.DataFrame(
        data=predicted_sentiment
    )
    predicted_sentiment_df.to_csv(predicted_sentiment_filename, index=False)


def load_embeddings(language):
    from fasttext.util import download_model
    from fasttext import load_model
    download_model(language, if_exists='ignore')
    embeddings = load_model('cc.' + language + '.300.bin')
    return embeddings


class Classifier:

    def __init__(self, manual_classes, language):
        self.embeddings = load_embeddings(language)
        self.size = len(self.embeddings[self.embeddings.words[0]])
        self.prototypes = {}
        for class_name, word_list in manual_classes.items():
            prototype_vector = np.zeros(self.size)
            for word in word_list:
                if word in self.embeddings:
                    prototype_vector += self.embeddings[word]
                else:
                    print("Word", word, "not in embeddings, skipping it")
            self.prototypes[class_name] = prototype_vector

    def classify(self, text):
        tokens = text.split()
        text_vector = np.zeros(self.size)
        for token in tokens:
            text_vector += self.embeddings[token]

        predictions = {}
        for class_name, prototype_vector in self.prototypes.items():
            # cosine similarity
            predictions[class_name] = np.dot(text_vector, prototype_vector) / (
                    np.linalg.norm(text_vector) * np.linalg.norm(prototype_vector)
            )

        return predictions


def predict(classifier, data_df):
    predictions = []
    texts = data_df.tolist()
    for text in tqdm(texts):
        predictions.append(classifier.classify(text))
    return predictions


def save_classes(predicted_classes, filename):
    df = pd.DataFrame(
        data=predicted_classes
    )
    df.to_csv(filename, index=False)


def remap_to_dict(remapped):
    d = {}
    for elem in remapped:
        groups = set(elem.keys())
        groups.remove("val")
        group_val_strings = []
        for group in sorted(list(groups)):
            group_val_strings.append("{}:{}".format(group, elem[group]))
        new_key = "__".join(group_val_strings)
        d[new_key] = elem
    return d
