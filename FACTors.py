# Class for FACTors dataset

import os, glob
import pandas as pd
import scripts.construction.preparation.stats as stats
import scripts.construction.preparation.data_cleaning.clean_dataset as clean_dataset

class FACTors():
    def __init__(self, path='./data/FACTors.csv'):
        """Initialises the dataset object and loads the dataset from the given path.

        Args:
            path (str, optional): Path of the dataset file, if available. Defaults to './data/FACTors.csv'.
        """
        self.name = 'FACTors'
        self.path = path
        self.data_folder = '<Enter the path to the raw scraped data folder here>'
        self.dataset = pd.read_csv(path)
        
    def __generate__(self):
        """Generates the dataset by concatenating all the CSV files in the raw data folder.
        """
        csv_files = glob.glob(os.path.join(self.data_folder + '/', "*.csv"))
        self.dataset = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)
        self.dataset.rename(columns={'rating': 'verdict'}, inplace=True)
        self.dataset = self.dataset.astype(str)
        
    def __clean__(self):
        """Clean the dataset by handling missing values, removing non-english articles, 
        correcting values, removing redundant phrases, converting smart quotes to dumb quotes, 
        and removing duplicates.
        """
        self.dataset = clean_dataset.handle_missing_values(self.dataset)
        self.dataset = clean_dataset.remove_non_english_articles(self.dataset)
        self.dataset = clean_dataset.correct_values(self.dataset)
        self.dataset = clean_dataset.remove_redundant_phrases(self.dataset)
        self.dataset = clean_dataset.convert_smart_quotes_to_dumb(self.dataset)
        self.dataset = clean_dataset.remove_duplicates(self.dataset)
        
    def __enumerate__(self):
        """Assigns unique IDs for each article and row in the dataset.
        """
        self.dataset['article_id'] = self.dataset['url'].astype('category').cat.codes
        self.dataset['row_id'] = range(1, len(self.dataset) + 1)
    
    def save(self, path='./data/FACTors.csv'):
        """Saves the dataset to the given path.

        Args:
            path (str, optional): Path where the dataset will be saved. Defaults to './data/FACTors.csv'.
        """
        self.dataset.to_csv(path, index=False)
        
    def load(self, path='./data/FACTors.csv'):
        """Loads the dataset from the given path.

        Args:
            path (str, optional): Path of the dataset file. Defaults to './data/FACTors.csv'.
        """
        self.dataset = pd.read_csv(path)
        
    def export_stats(self):
        """Print general statistics and generate author-organisation statistics for the dataset.
        """
        stats.print_general_stats(self.dataset)
        stats.generate_factchecker_stats(self.dataset, 'author')
        stats.generate_factchecker_stats(self.dataset, 'organisation')

