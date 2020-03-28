# design4emergency

## Setup

Clone repo:

```
git clone git@github.com:w4nderlust/design4emergency.git
```

Enter repo directory:

```
cd design4emergency
```

Create virtualenv:

```
virtualenv -p python3 venv
```

Enter in the virtualenv:

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Download spaCy languages (needed for lemmatization):

```
python -m spacy download it_core_news_sm
```

Install SentITA:

```
git clone https://github.com/w4nderlust/SentITA.git
```

Download https://drive.google.com/uc?id=1IN-RZL-gpgzuosr-BknKtA6reDJ48XNI

```
gdown https://drive.google.com/uc?id=1IN-RZL-gpgzuosr-BknKtA6reDJ48XNI
```

Move test_sentita_lstm-cnn_wikiner_v1.h5 inside SentITA/sentita folder

```
mv test_sentita_lstm-cnn_wikiner_v1.h5 SentITA/sentita
```

Install SentITA

```
pip install ./SentITA
```

Remove folder SentITA

```
rm -r SentITA
```

Download `data.tsv` and place it into the `data` folder

## Usage

Command:

```
python text_analysis.py data/data.tsv column_name column_name ...
```

Example:

```
python text_analysis.py data/data.tsv "Cosa ti fa più paura?" "Cosa ti fa stare bene?"
```

For more parameters check:

```
usage: text_analysis.py [-h] [-g GROUPS [GROUPS ...]] [-l LANGUAGE] [-lm]
                        [-nr NGRAM_RANGE] [-w NUM_WORDS] [-t NUM_TOPICS]
                        [-m MANUAL_MAPPINGS] [-wc WORD_CLOUD_FILENAME]
                        [-fw FREQUENT_WORDS_FILENAME]
                        [-fwp FREQUENT_WORDS_PLOT_FILENAME]
                        [-ttw TOP_TFIDF_WORDS_FILENAME]
                        [-ttwp TOP_TFIDF_WORDS_PLOT_FILENAME]
                        [-tp TOPICS_FILENAME] [-pt PREDICTED_TOPICS_FILENAME]
                        [-lv LDAVIS_FILENAME_PREFIX] [-s]
                        [-ps PREDICTED_SENTIMENT_FILENAME] [-o OUTPUT_PATH]
                        data_path columns [columns ...]

This script analyzes text in columns of a TSV file

positional arguments:
  data_path             path to the data TSV
  columns               columns to extract from TSV

optional arguments:
  -h, --help            show this help message and exit
  -g GROUPS [GROUPS ...], --groups GROUPS [GROUPS ...]
                        columns from the TSV to use for grouping
  -l LANGUAGE, --language LANGUAGE
                        language of the text in the data (for data cleaning
                        purposes)
  -lm, --lemmatize      performs lemmatization of all texts
  -nr NGRAM_RANGE, --ngram_range NGRAM_RANGE
                        minimum and maximum value for ngrams, specify as
                        "min,max"
  -w NUM_WORDS, --num_words NUM_WORDS
                        number of most frequent words to show
  -t NUM_TOPICS, --num_topics NUM_TOPICS
                        number of topics for topic modeling
  -m MANUAL_MAPPINGS, --manual_mappings MANUAL_MAPPINGS
                        path to JSON file contaning manual mappings
  -wc WORD_CLOUD_FILENAME, --word_cloud_filename WORD_CLOUD_FILENAME
                        path to save the word cloud to
  -fw FREQUENT_WORDS_FILENAME, --frequent_words_filename FREQUENT_WORDS_FILENAME
                        path to save frequent words to
  -fwp FREQUENT_WORDS_PLOT_FILENAME, --frequent_words_plot_filename FREQUENT_WORDS_PLOT_FILENAME
                        path to save the frequent word plot to
  -ttw TOP_TFIDF_WORDS_FILENAME, --top_tfidf_words_filename TOP_TFIDF_WORDS_FILENAME
                        path to save top tfidf words to
  -ttwp TOP_TFIDF_WORDS_PLOT_FILENAME, --top_tfidf_words_plot_filename TOP_TFIDF_WORDS_PLOT_FILENAME
                        path to save the top tfidf word plot to
  -tp TOPICS_FILENAME, --topics_filename TOPICS_FILENAME
                        path to save frequent words to
  -pt PREDICTED_TOPICS_FILENAME, --predicted_topics_filename PREDICTED_TOPICS_FILENAME
                        path to save predicted LDA topics for each datapoint
                        to
  -lv LDAVIS_FILENAME_PREFIX, --ldavis_filename_prefix LDAVIS_FILENAME_PREFIX
                        path (prefix) to save LDA vis plot files to
  -s, --predict_sentiment
                        performs sentiment analysis (it is pretty slow)
  -ps PREDICTED_SENTIMENT_FILENAME, --predicted_sentiment_filename PREDICTED_SENTIMENT_FILENAME
                        path to save predicted sentiment for each datapoint to
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path that will contain all directories, one for each
                        column
```

## Outputs

The script outputs one directory for each column spacified, contaning the following files:

- `wordcloud_filename` (`wordcloud.png`): a PNG file of a wordcloud of the text in the specified column
- `frequent_words_filename` (`frequent_words.json`): a JSON file contaning all non-stopword words in the text in the specified column with their respective frequency
- `frequent_words_plot_filename` (`frequent_words.png`): a PNG file containing a plot of top k most frequent non-stopword words in the text of the column
- `top_tfidf_words_filename` (`top_tfidf_words.json`): a JSON file contaning all non-stopword words in the text in the specified column with their respective tfidf
- `top_tfidf_words_plot_filename` (`top_tfidf_words.png`): a PNG file containing a plot of top k top tfidf non-stopword words in the text of the column
- `topics_filename` (`topics.json`): a JSON file containing, for each topic, for each word, the probability of that word given that topic
- `predicted_topics_filename` (`predicted_topics.csv`): a CSV file containing one column for each topic, and for each row in the input TSV file, the probability that the text in the input files column belongs to each topic
- `ldavis_filename_prefix` (`ldavis_`):
  - `ldavis_filename_prefix_N`: N is the number of topics. This file is a Python pickle file contaning the data to generate an HTML LDA visualization
  - `ldavis_filename_prefix_N.html`: a HTML visualization of the LDA topic distribution
- `predicted_sentiment_filename` (`predicted_sentiment.csv`): a CSV file containing a positive and a negative column and for each row in the original TSV file independent probability values for positive and negative sentiment (two low probabilities means neutral, two high probability values mean ambivalent). This is returned only if `--predict_sentiment` is provided, as calculating the sentiment can be pretty slow)
- if groups is spcified:
  - `wordcloud_filename`, `frequent_words_filename`, `frequent_words_plot_filename`, `top_tfidf_words_filename` and `top_tfidf_words_plot_filename` will be repeated for each groups for each value, with a `[group]_[value]_` prefix
  - two additional JSON files per each group collecting word frequencies and tfidf for all values will be collected, namely `[group]_frequent_words.json` and `[group]_tfidf_words.json`.
