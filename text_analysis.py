import os

from db_utils import connect_db, upload_db

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import load_data, clean_data, remove_stopwords, lemmatize_text, apply_manual_mappings, \
    plot_word_cloud, get_count_vectorizer_and_transformed_data, get_tfidf_vectorizer_and_transformed_data, \
    most_frequent_words, save_words, plot_top_words, learn_topic_model, print_topics, save_topics, \
    save_predicted_topics, visualize_topic_model, predict_sentiment_with_paralleldots, predict_sentiment_with_sentita, \
    save_sentiment, remap_to_dict

import argparse
import unidecode
import json
import numpy as np
import pandas as pd


def text_analysis(
        data_path,
        column,
        groups,
        language,
        lemmatize,
        ngram_range,
        num_topics,
        num_words,
        manual_mappings,
        generate_word_cloud,
        word_cloud_filename,
        frequent_words_filename,
        frequent_words_plot_filename,
        top_tfidf_words_filename,
        top_tfidf_words_plot_filename,
        predict_topics,
        topics_filename,
        predicted_topics_filename,
        ldavis_filename_prefix,
        predict_sentiment,
        predicted_sentiment_filename,
        should_upload_db,
        account_key_path
):
    print("Loading data...")
    data_df = load_data(data_path, column, groups)
    print("Loaded data sample")
    print(data_df.head())
    print()

    print("Cleaning data...")
    data_df[column] = clean_data(data_df[column])
    print("Clean data sample")
    print(data_df.head())
    print()

    print("Removing stop words from data...")
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

    if generate_word_cloud:
        print("Generating word cloud...")
        plot_word_cloud(data_df[column], word_cloud_filename, language)
        print("word_cloud saved to:", word_cloud_filename)
        print()

    count_vectorizer, count_data = get_count_vectorizer_and_transformed_data(
        data_df[column], language, ngram_range
    )
    all_word_count_pair_list = most_frequent_words(
        count_data, count_vectorizer, count_data.shape[0] + 1
    )
    word_count_pair_list = all_word_count_pair_list[:num_words]

    tfidf_vectorizer, tfidf_data = get_tfidf_vectorizer_and_transformed_data(
        data_df[column], language, ngram_range
    )
    all_tfidf_pair_list = most_frequent_words(
        tfidf_data, tfidf_vectorizer, tfidf_data.shape[0] + 1
    )
    tfidf_pair_list = all_tfidf_pair_list[:num_words]

    print("Saving frequent words...")
    save_words(
        all_word_count_pair_list,
        frequent_words_filename
    )
    print("Frequent words saved to:", frequent_words_filename)
    print()

    if should_upload_db:
        db_client = connect_db(account_key_path)
    else:
        db_client = None

    if should_upload_db:
        print("Uploading frequent words to db...")
        upload_db(db_client, 'frequent_words', {
            column: {w: int(c) for w, c in word_count_pair_list}
        })
        print('Done')
        print()

    print("Generating frequent word plot...")
    plot_top_words(word_count_pair_list, frequent_words_plot_filename)
    print("Frequent word plot saved to:", frequent_words_plot_filename)
    print()

    print("Saving top tfidf words...")
    save_words(
        all_tfidf_pair_list,
        top_tfidf_words_filename
    )
    print("Top tfidf words saved to:", top_tfidf_words_filename)
    print()

    if should_upload_db:
        print("Uploading frequent words to db...")
        upload_db(db_client, 'top_tfidf', {
            column: {w: int(c) for w, c in tfidf_pair_list}
        })
        print('Done')
        print()

    print("Generating top tfidf word plot...")
    plot_top_words(tfidf_pair_list, top_tfidf_words_plot_filename)
    print("Top tfidf word plot saved to:", top_tfidf_words_plot_filename)
    print()

    if groups:
        group_unique_vals = {}
        for group in groups:
            group_unique_vals[group] = data_df[group].unique()

        splits = {}
        for group, unique_vals in group_unique_vals.items():
            for val in unique_vals:
                splits[(group, val)] = data_df[group] == val

        for i in range(len(groups) - 1):
            splits = concat_splits(splits)

        grouped_words_counts = {}
        grouped_words_tfidf = {}

        for key, split_idcs in splits.items():
            split = data_df[split_idcs]
            split_texts = split[column]

            if len(split_texts) > 0 and any(split_texts.str.len() > 0):
                word_cloud_filename_val = add_prefix_to_filename(
                    word_cloud_filename, key
                )
                frequent_words_filename_val = add_prefix_to_filename(
                    frequent_words_filename, key
                )
                frequent_words_plot_filename_val = add_prefix_to_filename(
                    frequent_words_plot_filename, key
                )
                top_tfidf_words_filename_val = add_prefix_to_filename(
                    top_tfidf_words_filename, key
                )
                top_tfidf_words_plot_filename_val = add_prefix_to_filename(
                    top_tfidf_words_plot_filename, key
                )

                if generate_word_cloud:
                    print("Generating word cloud...")
                    plot_word_cloud(split_texts, word_cloud_filename_val, language)
                    print("word_cloud saved to:", word_cloud_filename_val)
                    print()

                try:
                    count_vectorizer, count_data = get_count_vectorizer_and_transformed_data(
                        split_texts, language, ngram_range
                    )
                    all_word_count_pair_list = most_frequent_words(
                        count_data, count_vectorizer, count_data.shape[0] + 1
                    )
                    word_count_pair_list = all_word_count_pair_list[:num_words]

                    tfidf_vectorizer, tfidf_data = get_tfidf_vectorizer_and_transformed_data(
                        split_texts, language, ngram_range
                    )
                    all_tfidf_pair_list = most_frequent_words(
                        tfidf_data, tfidf_vectorizer, tfidf_data.shape[0] + 1
                    )
                    tfidf_pair_list = all_tfidf_pair_list[:num_words]

                    print("Saving frequent words...")
                    save_words(
                        all_word_count_pair_list,
                        frequent_words_filename_val
                    )
                    print("Frequent words saved to:", frequent_words_filename_val)
                    print()

                    print("Generating frequent word plot...")
                    plot_top_words(word_count_pair_list, frequent_words_plot_filename_val)
                    print("Frequent word plot saved to:", frequent_words_plot_filename_val)
                    print()

                    print("Saving top tfidf words...")
                    save_words(
                        all_tfidf_pair_list,
                        top_tfidf_words_filename_val
                    )
                    print("Top tfidf words saved to:", top_tfidf_words_filename_val)
                    print()

                    print("Generating top tfidf word plot...")
                    plot_top_words(tfidf_pair_list, top_tfidf_words_plot_filename_val)
                    print("Top tfidf word plot saved to:", top_tfidf_words_plot_filename_val)
                    print()

                    grouped_words_counts[key[1::2]] = {
                        w: int(c) for w, c in all_word_count_pair_list
                    }
                    grouped_words_tfidf[key[1::2]] = {
                        w: int(c) for w, c in all_tfidf_pair_list
                    }
                except:
                    print("Error processing", key,
                          "skipping it. texts are probably all stopwords")

        print("Saving grouped frequent words...")
        group_frequent_words_filename = add_prefix_to_filename(
            frequent_words_filename, groups
        )
        remapped_grouped_words_counts = remap_keys(grouped_words_counts, groups)
        with open(group_frequent_words_filename, 'w', encoding="utf8") as f:
            json.dump(remapped_grouped_words_counts, f, ensure_ascii=False)
        print("Frequent words saved to:", group_frequent_words_filename)
        print()

        if should_upload_db:
            print("Uploading grouped_words_counts to db...")
            upload_db(db_client, 'grouped_words_counts', {
                column: remap_to_dict(remapped_grouped_words_counts)
            })
            print('Done')
            print()

        print("Saving grouped top tfidf words...")
        group_top_tfidf_words_filename = add_prefix_to_filename(
            top_tfidf_words_filename, groups
        )
        remapped_grouped_words_tfidf = remap_keys(grouped_words_tfidf, groups)
        with open(group_top_tfidf_words_filename, 'w', encoding="utf8") as f:
            json.dump(remapped_grouped_words_tfidf, f, ensure_ascii=False)
        print("Top tfidf words saved to:", group_top_tfidf_words_filename)
        print()

        if should_upload_db:
            print("Uploading grouped_words_tfidf to db...")
            upload_db(db_client, 'grouped_words_tfidf', {
                column: remap_to_dict(remapped_grouped_words_tfidf)
            })
            print('Done')
            print()

    if predict_topics:
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

        if should_upload_db:
            print("Uploading predicted topics to db...")
            upload_db(db_client, 'predicted_topics', {
                column: json.loads(pd.DataFrame(predicted_topics).to_json(
                    orient='index', force_ascii=False
                ))
            })
            print('Done')
            print()

        print("Generating LDA visualization...")
        visualize_topic_model(lda, count_data, tfidf_vectorizer,
                              num_topics, ldavis_filename_prefix)
        print("LDA visualization saved to:", ldavis_filename_prefix)
        print()

    if predict_sentiment:
        if language == 'it':
            print("Predict sentiment...")
            predicted_sentiment = predict_sentiment_with_sentita(data_df[column])
            save_sentiment(predicted_sentiment, predicted_sentiment_filename)
            print("Predict sentiment saved to:", predicted_sentiment_filename)
            print()

            if should_upload_db:
                print("Uploading predicted sentiment to db...")
                upload_db(db_client, 'predicted_sentiment', {
                    column: json.loads(pd.DataFrame(predicted_sentiment).to_json(
                        orient='index', force_ascii=False
                    ))
                })
                print('Done')
                print()

        elif language == 'en':
            print("Predict sentiment...")
            predicted_sentiment = predict_sentiment_with_paralleldots(data_df)
            save_sentiment(predicted_sentiment, predicted_sentiment_filename)
            print("Predict sentiment saved to:", predicted_sentiment_filename)
            print()

            if should_upload_db:
                print("Uploading predicted sentiment to db...")
                upload_db(db_client, 'predicted_sentiment', {
                    column: json.loads(pd.DataFrame(predicted_sentiment).to_json(
                        orient='index', force_ascii=False
                    ))
                })
                print('Done')
                print()
        else:
            print("Sentiment analysis on {} language is not supported")
            print()


#######
# Utils
#######
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
    filename = unidecode.unidecode(s)
    filename = ''.join(c for c in filename if c in valid_chars)
    filename = filename.replace(' ', '_')
    filename = filename.lower()
    return filename


def add_prefix_to_filename(fn_path, prefix):
    split_path = list(os.path.split(fn_path))
    if isinstance(prefix, (list, tuple)):
        prefix = "_".join(prefix)
    split_path[-1] = format_filename(prefix) + "_" + split_path[-1]
    return os.path.join(*split_path)


def concat_splits(splits):
    # splits[(group, val)] = split
    c_splits = {}
    already_added = set()
    for key, split in splits.items():
        for o_key, o_split in splits.items():
            if len(o_key) == 2:  # only group and val
                if len(key) == 2 and key[0] == o_key[0] and key[1] == o_key[1]:
                    c_splits[key] = split
                    already_added.add("_".join(sorted(key)))
                elif o_key[0] not in set(key[::2]):
                    new_key = key + o_key
                    new_key_sorted = "_".join(sorted(new_key))
                    if new_key_sorted not in already_added:
                        c_splits[new_key] = np.logical_and(
                            split, o_split
                        )
                        already_added.add(new_key_sorted)
    return c_splits


def populate(dict, splits):
    for key in splits:
        prev_dict = dict
        # only even indices elements, meaning only unique vals, not group names
        for val in key[1::2]:
            new_dict = prev_dict.get(val, {})
            prev_dict[val] = new_dict
            prev_dict = new_dict
    return dict


def remap_keys(mapping, groups):
    entries = []
    for k, v in mapping.items():
        entry = {}
        for i in range(len(k)):
            entry[groups[i]] = k[i]
        entry['val'] = v
        entries.append(entry)
    return entries


# def remap_keys_nested(mapping):
#     nested = {}
#     for k, v in mapping.items():
#         prev_dict = nested
#         for g in k:
#             if g not in prev_dict:
#                 prev_dict[g] = {}
#             prev_dict = prev_dict[g]
#         prev_dict.update(v)
#     return nested
#
# def remap_keys_slash(mapping):
#     return {'/'.join(k): v for k, v in mapping.items()}

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
        '-g',
        '--groups',
        type=str,
        nargs='+',
        help='columns from the TSV to use for grouping'
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
        '-m',
        '--manual_mappings',
        type=str,
        help='path to JSON file contaning manual mappings',
        default=None
    )
    parser.add_argument(
        '-gw',
        '--generate_word_cloud',
        action='store_true',
        help='generates word cloud plots',
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
        '-pt',
        '--predict_topics',
        action='store_true',
        help='learns topics and predicts them for each text (pretty slow)',
    )
    parser.add_argument(
        '-tf',
        '--topics_filename',
        type=str,
        help='path to save frequent words to',
        default='topics.json'
    )
    parser.add_argument(
        '-ptf',
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
    parser.add_argument(
        '-u',
        '--should_upload_db',
        action='store_true',
        help='uploads to db',
    )
    parser.add_argument(
        '-akp',
        '--account_key_path',
        type=str,
        help='path to che account key JSON file',
        default=''
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
        column_dir = os.path.join(args.output_path, format_filename(column))
        if not os.path.exists(column_dir):
            os.makedirs(column_dir)

        word_cloud_filename = os.path.join(
            column_dir, args.word_cloud_filename
        )
        frequent_words_filename = os.path.join(
            column_dir, args.frequent_words_filename
        )
        frequent_words_plot_filename = os.path.join(
            column_dir, args.frequent_words_plot_filename
        )
        top_tfidf_words_filename = os.path.join(
            column_dir, args.top_tfidf_words_filename
        )
        top_tfidf_words_plot_filename = os.path.join(
            column_dir, args.top_tfidf_words_plot_filename
        )
        topics_filename = os.path.join(
            column_dir, args.topics_filename
        )
        predicted_topics_filename = os.path.join(
            column_dir, args.predicted_topics_filename
        )
        ldavis_filename_prefix = os.path.join(
            column_dir, args.ldavis_filename_prefix
        )
        predicted_sentiment_filename = os.path.join(
            column_dir, args.predicted_sentiment_filename
        )

        text_analysis(
            data_path=args.data_path,
            column=column,
            groups=args.groups,
            language=args.language,
            lemmatize=args.lemmatize,
            ngram_range=ngram_range,
            num_topics=args.num_topics,
            num_words=args.num_words,
            manual_mappings=args.manual_mappings,
            generate_word_cloud=args.generate_word_cloud,
            word_cloud_filename=word_cloud_filename,
            frequent_words_filename=frequent_words_filename,
            frequent_words_plot_filename=frequent_words_plot_filename,
            top_tfidf_words_filename=top_tfidf_words_filename,
            top_tfidf_words_plot_filename=top_tfidf_words_plot_filename,
            predict_topics=args.predict_topics,
            topics_filename=topics_filename,
            predicted_topics_filename=predicted_topics_filename,
            ldavis_filename_prefix=ldavis_filename_prefix,
            predict_sentiment=args.predict_sentiment,
            predicted_sentiment_filename=predicted_sentiment_filename,
            should_upload_db=args.should_upload_db,
            account_key_path=args.account_key_path,
        )
