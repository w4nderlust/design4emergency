import argparse
import json
import os

from text_analysis import format_filename
from utils import load_data, clean_data, remove_stopwords, lemmatize_text, \
    apply_manual_mappings, Classifier, predict, save_classes


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
    data_df[column] = remove_stopwords(data_df[column], language)
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
