# General statistics about the dataset
def percentage_articles_with_signature(dataset_pd_obj):
    return dataset_pd_obj[dataset_pd_obj['author'] != 'Unknown'].shape[0] / len(dataset_pd_obj) * 100

def percentage_articles_with_multiple_authors(dataset_pd_obj):
    return dataset_pd_obj[dataset_pd_obj['author'].str.contains(';')].shape[0] / len(dataset_pd_obj) * 100

def number_of_unique_articles(dataset_pd_obj):
    return dataset_pd_obj['url'].nunique()

def number_of_unique_claims(dataset_pd_obj):
    return dataset_pd_obj['claim_id'].nunique()

def number_of_overlapping_claims(dataset_pd_obj):
    value_counts = dataset_pd_obj['claim_id'].value_counts()
    return (value_counts > 1).sum(), value_counts[value_counts > 1].sum()

def average_word_count(dataset_pd_obj):
    return dataset_pd_obj['content'].apply(lambda x: len(str(x).split())).mean()

def print_general_stats(dataset_pd_obj):
    print("The size of the dataset is", len(dataset_pd_obj))
    print("The number of unique articles is", number_of_unique_articles(dataset_pd_obj))
    print("The number of unique claims is", number_of_unique_claims(dataset_pd_obj))
    num_overlapping, num_rows = number_of_overlapping_claims(dataset_pd_obj)
    print(f"The number of overlapping claims is {num_overlapping}, covering {num_rows} fact-checks")
    print("The average word count of articles is", average_word_count(dataset_pd_obj))
    print("The percentage of articles with a signature is", percentage_articles_with_signature(dataset_pd_obj))
    print("The percentage of articles with multiple authors is", percentage_articles_with_multiple_authors(dataset_pd_obj))
    
