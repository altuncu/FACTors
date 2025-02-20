# Helper functions to clean the dataset

import langdetect
import re
import pandas as pd
from nltk import word_tokenize
from collections import Counter
from itertools import islice
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining
from datetime import datetime

model = SentenceTransformer("all-MiniLM-L6-v2")

def remove_non_english_articles(dataset):
    """Removes the non-English articles from a given dataframe.

    Args:
        dataset (pd.DataFrame): Dataframe containing the articles.

    Returns:
        pd.DataFrame: Cleaned dataframe with only English articles.
    """
    print("Removing non-English articles from the dataset with size", len(dataset))
    dataset['language_c'] = dataset['content'].apply(detect_language)
    print(len(dataset[(dataset['language_c'] == 'en')]))
    dataset['language_t'] = dataset['title'].apply(detect_language)
    print(len(dataset[(dataset['language_t'] == 'en')]))
    dataset = dataset[(dataset['language_c'] == 'en') & (dataset['language_t'] == 'en')]
    dataset = dataset.drop(columns=['language_c', 'language_t'])
    print("The size of the dataset after removing non-English articles is", len(dataset))
    return dataset

def handle_missing_values(dataset):
    """Handles the missing values in the dataset.

    Args:
        dataset (pd.DataFrame): Dataframe containing the articles.

    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    print("Removing missing values from the dataset with size", len(dataset))
    dataset['content'] = dataset['content'].apply(handle_missing_content)
    print("Content without any letters have been removed: ", len(dataset))
    dataset['claim'] = dataset.apply(lambda row: row['title'] 
                                     if len(str(row['claim'])) < 10 or row['claim'] == 'No claim found' or not row['claim'].strip()
                                     else row['claim'], axis=1)
    dataset['author'] = dataset.apply(lambda row: 'Unknown' 
                                     if not str(row['author']).strip() 
                                     else row['author'], axis=1)
    dataset = dataset.dropna(subset=['title', 'content', 'url', 'date_published', 'claim', 'verdict'])
    print("Empty values have been removed: ", len(dataset))
    print("The size of the dataset after removing missing values is", len(dataset))
    return dataset

def handle_missing_content(c):
    """Removes the articles with too short content.

    Args:
        c (float): Column value with the content of the article.

    Returns:
        string: Content of the article if it has more than 10 characters.
    """
    if re.compile("[a-z]+", re.I).search(str(c)) and len(str(c)) > 10:
        return c
    else:
        return None
    
def detect_language(c):
    """Detects the language of the given text.

    Args:
        c (float): Input text from dataframe.

    Returns:
        _type_: Language of the text if it is detected, otherwise '?'.
    """
    try:
        return langdetect.detect(str(c))
    except:
        return '?'
    
def handle_dates(c):
    """Handles dates in different formats.

    Args:
        c (float): Date value from the dataset

    Returns:
        string: Date in the format of 'YYYY-MM-DDTHH:MM:SS'.
    """
    if len(str(c)) == 14:
        return c[:4] + '-' + c[5:7] + '-' + c[10:12]
    elif 'Published' in str(c):
        if 'Updated' in str(c):
            date = c.split('Updated')[0].replace('Published', '').replace('.', '').strip()
        else:
            date = c.replace('Published', '').replace('.', '').strip()
        if date[0] == ':':
            date = date[1:].strip()
        date = ' '.join(date.split(' ')[:2] + date.split(' ')[3:])
        return datetime.strptime(date, '%I:%M %p %b %d, %Y').strftime('%Y-%m-%dT%H:%M:%S')
    else:
        return c
    
def correct_values(dataset):
    """Standardises the values in the dataset.

    Args:
        dataset (pd.DataFrame): Dataframe containing the articles.

    Returns:
        pd.DataFrame: Dataframe with standardised values.
    """
    dataset['date_published'] = dataset['date_published'].apply(handle_dates)
    dataset['date_published'] = pd.to_datetime(dataset['date_published'], format='mixed', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S')
    dataset['author'] = dataset['author'].apply(
                            lambda x: ';'.join([item.strip() for item in str(x).replace('"', '').split(',') if item.strip()])
                            )
    dataset['claim'] = dataset['claim'].apply(lambda c: re.sub(r'\[.*?\]', '', str(c)).strip())
    dataset['claim'] = dataset['claim'].apply(lambda c: re.sub(r'\(.*?\)', '', str(c)).strip())
    dataset['verdict'] = dataset['verdict'].apply(lambda c: re.sub(r'\[.*?\]', '', str(c)).strip())
    dataset['verdict'] = dataset['verdict'].apply(lambda c: re.sub(r'\(.*?\)', '', str(c)).strip())
    dataset['verdict'] = dataset['verdict'].apply(lambda c: str(c).title())
    return dataset

def most_common_ngrams(dataset, min_n=3, max_n=6):
    """Retrieves 500 most common n-grams from the dataset to look for redundant phrases.

    Args:
        dataset (pd.DataFrame): Dataframe containing the articles.
        min_n (int, optional): Minimum n-gram length. Defaults to 3.
        max_n (int, optional): Maximum n-gram length. Defaults to 6.
    """
    ngram_counter = Counter()
    
    claims = dataset['claim'].tolist()
    for text in claims:
        words = word_tokenize(str(text).lower())
    
        for n in range(min_n, max_n + 1):
            ngrams = zip(*(islice(words, i, None) for i in range(n)))
            ngram_counter.update([' '.join(ngram) for ngram in ngrams])
    
    most_common_ngrams = ngram_counter.most_common(500)
    for ngram, count in most_common_ngrams:
        print(f"{ngram} -> {count}")

def remove_redundant_phrases(dataset):
    """Removes the redundant phrases given in the text file, from the dataset.

    Args:
        dataset (pd.DataFrame): Dataframe containing the articles.

    Returns:
        pd.DataFrame: Dataframe with redundant phrases removed.
    """
    with open('data/redundant_phrases.txt', 'r', encoding='utf-16') as f:
        redundant_phrases = f.readlines()
        pattern = '|'.join(re.escape(phrase.rstrip()) for phrase in redundant_phrases)
        dataset['claim'] = dataset['claim'].apply(lambda c: re.sub(pattern, '', str(c), flags=re.IGNORECASE).strip())
        dataset['verdict'] = dataset['verdict'].apply(lambda c: re.sub(pattern, '', str(c), flags=re.IGNORECASE).strip())
        return dataset
    
def convert_smart_quotes_to_dumb(dataset):
    """Convert smart quotes to dumb quotes in the dataset.

    Args:
        dataset (pd.DataFrame): Dataframe containing the articles.

    Returns:
        pd.DataFrame: Dataframe with smart quotes converted to dumb quotes.
    """
    smart_to_dumb = {
        '“': '"', '”': '"',  
        '‘': "'", '’': "'"  
    }
    for column in ['claim', 'verdict', 'content', 'title']:
        dataset[column] = dataset[column].apply(
            lambda x: re.sub(
                '|'.join(re.escape(k) for k in smart_to_dumb.keys()), 
                lambda m: smart_to_dumb[m.group()], 
                str(x) if isinstance(x, str) else x
            )
        )
    return dataset

def remove_duplicates(dataset_pd_obj):
    """Removes the duplicate articles from the dataset based on the similarity of claims
    from the same organisation.

    Args:
        dataset (pd.DataFrame): Dataframe containing the articles.

    Returns:
        pd.DataFrame: Dataframe with duplicate articles removed.
    """
    claims = dataset_pd_obj['claim'].to_list()
    lookup_table_org = dict(zip(dataset_pd_obj['claim'], dataset_pd_obj['organisation']))
    similarity_scores = paraphrase_mining(model, claims, show_progress_bar=True, top_k=100)
    similarity_scores = [item for item in similarity_scores 
                         if item[0] > 0.95 and 
                         lookup_table_org.get(claims[item[1]]) == lookup_table_org.get(claims[item[2]])]
    for c in similarity_scores:
        dataset_pd_obj = dataset_pd_obj[dataset_pd_obj['claim'] != claims[c[1]]]
    return dataset_pd_obj