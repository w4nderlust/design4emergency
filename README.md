# design4emergency


## Setup

Clone repo:
```
git clone git@github.com:tezzutezzu/design4emergency.git
```

Enter repo directory:
```
cd design4emergency
```

Create virtualenv:
```
viertualenv -p python3 venv
```

Enter in the virtualenv:
```
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Command:
```
python text_analysis.py filename/to/dataset column_name column_name ...
```

Example:
```
python text_analysis.py "/home/piero/data/DfE_Dataset_Corriere_1600_Danilo - Form Responses 1.tsv" "Cosa ti fa più paura?" "Cosa ti fa stare bene?"
```

For more parameters check:
```
python text_analysis.py  -h
```

```
usage: ludwig train [options]

This script trains a model

positional arguments:
  data_path             path to the data TSV
  columns               columns to extract from TSV

optional arguments:
  -h, --help            show this help message and exit
  -l LANGUAGE, --language LANGUAGE
                        language of the text in the data (for data cleaning
                        purposes)
  -w NUM_WORDS, --num_words NUM_WORDS
                        number of most frequent words to show
  -t NUM_TOPICS, --num_topics NUM_TOPICS
                        number of topics for topic modeling
  -wc WORDCLOUD_FILENAME, --wordcloud_filename WORDCLOUD_FILENAME
                        path to save the wordcloud to
  -fw FREQUENT_WORDS_FILENAME, --frequent_words_filename FREQUENT_WORDS_FILENAME
                        path to save frequent words to
  -fwp FREQUENT_WORDS_PLOT_FILENAME, --frequent_words_plot_filename FREQUENT_WORDS_PLOT_FILENAME
                        path to save the frequent word plot to
  -tp TOPICS_FILENAME, --topics_filename TOPICS_FILENAME
                        path to save frequent words to
  -pt PREDICTED_TOPICS_FILENAME, --predicted_topics_filename PREDICTED_TOPICS_FILENAME
                        path to save predicted LDA topics for each datapoint
                        to
  -lv LDAVIS_FILENAME_PREFIX, --ldavis_filename_prefix LDAVIS_FILENAME_PREFIX
                        path (prefix) to save LDA vis plot files to
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path that will contain all directories, one for each
                        column
```

## Outputs

The script outputs one directory for each column spacified, contaning the following files:

- `wordcloud_filename` (`wordcloud.png`): a PNG file of a wordcloud of the text in the specified column
- `frequent_words_filename` (`frequent_words.json`): a JSON file contaning all non-stopword words in the text in the specified column with their respective frequency
- `frequent_words_plot_filename` (`frequent_words.png`): a PNG file containing a plot of top k most frequent non-stopword words in the text of the column
- `topics_filename` (`topics.json`): a JSON file containing, for each topic, for each word, the probability of that word given that topic
- `predicted_topics_filename` (`predicted_topics.csv`): a CSV file containing one column for each topic, and for each row in the input TSV file, the robability that the text in the input files column belongs to each topic
- `ldavis_filename_prefix` (`ldavis_`):
  - `ldavis_filename_prefix_N`: N is the number of topics. This file is a Python pickle file contaning the data to generate an HTML LDA visualization
  - `ldavis_filename_prefix_N.html`: a HTML visualization of the LDA topic distribution
