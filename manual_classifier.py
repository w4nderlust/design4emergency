import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from text_analysis import load_data, clean_data, lemmatize_text, apply_manual_mappings, format_filename, \
    remove_stopwords
from text_analysis import stopwords as all_stopwords

stopwords = set(all_stopwords['it'])


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


#########
# predict
#########
def predict(classifier, data_df):
    predictions = []
    texts = data_df.tolist()
    for text in tqdm(texts):
        predictions.append(classifier.classify(text))
    return predictions


##############
# save classes
##############
def save_classes(predicted_classes, filename):
    df = pd.DataFrame(
        data=predicted_classes
    )
    df.to_csv(filename, index=False)


def manual_classes_classifier(
        data_path,
        column,
        language,
        lemmatize,
        manual_mappings,
        manual_classes,
        predicted_classes_filename
):
    print("Build classifier...")
    with open(manual_classes, encoding="utf8") as json_data:
        manual_classes_dict = json.load(json_data)
    classifier = Classifier(manual_classes_dict, language)
    print("Classifier built")
    print()

    print("Loading data...")
    data_df = load_data(data_path, column)
    print("Loaded data sample")
    print(data_df.head())
    print()

    print("Cleaning data...")
    data_df[column] = clean_data(data_df[column])
    print("Clean data sample")
    print(data_df.head())
    print()

    print("Removing stopwors...")
    data_df = remove_stopwords(data_df, language)
    print("Data sample")
    print(data_df.head())
    print()

    if lemmatize:
        print("Lemmatizing data...")
        data_df[column] = lemmatize_text(data_df[column], language)
        print("Lemmatized data sample")
        print(data_df.head())
        print()

    if manual_mappings:
        print("Applying manual mappings...")
        data_df[column] = apply_manual_mappings(data_df[column], manual_mappings)
        print("Manually mapped data sample")
        print(data_df.head())
        print()

    print("Predict classes...")
    predicted_classes = predict(classifier, data_df[column])
    save_classes(predicted_classes, predicted_classes_filename)
    print("Predicted classes saved to:", predicted_classes_filename)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script creates a classifier from a bag of words '
                    'using embeddings and uses it to predict classes'
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
        '-mc',
        '--manual_classes',
        type=str,
        help='path to JSON file contaning manual classes',
        default=None,
        required=True
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
        '-mm',
        '--manual_mappings',
        type=str,
        help='path to JSON file contaning manual mappings',
        default=None
    )
    parser.add_argument(
        '-pc',
        '--predicted_classes_filename',
        type=str,
        help='path to save predicted classes for each datapoint to',
        default='predicted_classes.csv'
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

        predicted_classes_filename = os.path.join(
            args.output_path, column_dir, args.predicted_classes_filename
        )

        manual_classes_classifier(
            args.data_path,
            column,
            args.language,
            args.lemmatize,
            args.manual_mappings,
            args.manual_classes,
            predicted_classes_filename
        )
