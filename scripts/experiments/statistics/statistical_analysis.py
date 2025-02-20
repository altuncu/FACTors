import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_trends_diagram(dataset_pd_obj):
    """Plot the graph of the number of articles and organisations over the years.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object containing the data.
    """
    dataset_pd_obj['year'] = dataset_pd_obj['date_published'].str[:4]
    dataset_pd_obj = dataset_pd_obj[dataset_pd_obj['year'] != "2025"]
    
    yearly_counts = dataset_pd_obj.groupby('year')['url'].nunique()
    yearly_org_counts = dataset_pd_obj.groupby('year')['organisation'].nunique()
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Articles', color='b')
    ax1.plot(np.asarray(yearly_counts.index, float), yearly_counts.values, color='b', marker='.')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Organisations', color='r')
    ax2.plot(np.asarray(yearly_org_counts.index, float), yearly_org_counts.values, color='r', marker='.')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    plt.savefig("trends.pdf", format="pdf", bbox_inches="tight")
    plt.show()



# Author and organisation statistics
def first_to_last_fact_check(dataset_pd_obj, column_grouped_by):
    """Time between the first and last fact-check for each author or organisation.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object containing the data.
        column_grouped_by (str): 'author' or 'organisation'.

    Returns:
        pd.Dataframe: List of authors or organisation with their experience in days.
    """
    if column_grouped_by == 'author':
        dataset_pd_obj = dataset_pd_obj.assign(author=dataset_pd_obj['author'].str.split(';')).explode('author')
    dataset_pd_obj['date_published'] = pd.to_datetime(dataset_pd_obj['date_published'])
    stats = dataset_pd_obj.groupby(column_grouped_by).agg(
        max_date=('date_published', 'max'),
        min_date=('date_published', 'min')
    )
    stats['experience_days'] = (stats['max_date'] - stats['min_date']).dt.days
    return stats['experience_days'].reset_index()

    
def number_of_fact_checks(dataset_pd_obj, column_grouped_by):
    """Total number of fact-checks for each author or organisation.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object containing the data.
        column_grouped_by (str): 'author' or 'organisation'.

    Returns:
        pd.Dataframe: List of authors or organisation with their number of fact-checks.
    """
    if column_grouped_by == 'author':
        dataset_pd_obj = dataset_pd_obj.assign(author=dataset_pd_obj['author'].str.split(';')).explode('author')
    stats = dataset_pd_obj.groupby(column_grouped_by)['row_id'].nunique()
    return stats.reset_index().rename(columns={'row_id': 'number_of_fact_checks'})

def percentage_unique_fact_checks(dataset_pd_obj, column_grouped_by):
    """Percentage of unique fact-checks, not fact-checked by any other author or organisation.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object containing the data.
        column_grouped_by (str): 'author' or 'organisation'.

    Returns:
        pd.Dataframe: List of authors or organisation with their unique fact-check percentage.
    """
    if column_grouped_by == 'author':
        dataset_pd_obj = dataset_pd_obj.assign(author=dataset_pd_obj['author'].str.split(';')).explode('author')
    id_sets = dataset_pd_obj.groupby(column_grouped_by)['claim_id'].apply(set)
    unique_id_counts = {}
    for name, ids in id_sets.items():
        other_ids = set().union(*[s for n, s in id_sets.items() if n != name])
        unique_ids = ids - other_ids
        unique_id_counts[name] = len(unique_ids) / len(ids) if len(ids) > 0 else 0
    return pd.DataFrame(list(unique_id_counts.items()), columns=[column_grouped_by, 'unique_claim_percentage'])

def calculate_freq_stats(group):
    """Fact-checking frequency statistics for each author or organisation.

    Args:
        group (pd.DataFrame): Grouped dataset object containing dates of fact-checks.

    Returns:
        pd.Series: Mean and standard deviation of differences between consecutive fact-check dates.
    """
    date_diffs = group.sort_values().diff().dropna().dt.days
    return pd.Series({
        'fc_frequency_mean': date_diffs.mean(),
        'fc_frequency_std': date_diffs.std()
    })

def fact_checking_rate(dataset_pd_obj, column_grouped_by):
    """Fact-checking frequency statistics for each author or organisation.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object containing the data.
        column_grouped_by (str): 'author' or 'organisation'.

    Returns:
        pd.Dataframe: List of authors or organisation with their fact-checking rates.
    """
    if column_grouped_by == 'author':
        dataset_pd_obj = dataset_pd_obj.assign(author=dataset_pd_obj['author'].str.split(';')).explode('author')
    dataset_pd_obj['date_published'] = pd.to_datetime(dataset_pd_obj['date_published'])
    stats = dataset_pd_obj.groupby(column_grouped_by)['date_published'].apply(calculate_freq_stats).reset_index()
    stats = stats.pivot(index=column_grouped_by, columns='level_1', values='date_published').reset_index()
    stats.columns.name = None
    return stats

def number_of_fact_checkers(dataset_pd_obj, column_grouped_by):
    """Number of authors each organisation has or 
    number of organisations each author has fact-checked for.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object containing the data.
        column_grouped_by (str): 'author' or 'organisation'.

    Returns:
        pd.Dataframe: List of authors or organisation with their number of associated authors/organisations.
    """
    if column_grouped_by == 'author':
        dataset_pd_obj = dataset_pd_obj.assign(author=dataset_pd_obj['author'].str.split(';')).explode('author')
        fact_checker = 'organisation'
    elif column_grouped_by == 'organisation':
        fact_checker = 'author'
    stats = dataset_pd_obj.groupby(column_grouped_by)[fact_checker].nunique().reset_index()
    return stats.rename(columns={fact_checker: 'number_of_' + fact_checker + 's'})

def word_count_stats(dataset_pd_obj, column_grouped_by):
    """Word count statistics for each author or organisation.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object containing the data.
        column_grouped_by (str): 'author' or 'organisation'.

    Returns:
        pd.Dataframe: List of authors or organisation with their average word counts.
    """
    if column_grouped_by == 'author':
        dataset_pd_obj = dataset_pd_obj.assign(author=dataset_pd_obj['author'].str.split(';')).explode('author')
    dataset_pd_obj['word_count'] = dataset_pd_obj['content'].apply(lambda x: len(x.split()))
    stats = dataset_pd_obj.groupby(column_grouped_by)['word_count'].agg(['mean', 'std']).reset_index()
    stats.rename(columns={'mean': 'word_count_mean', 'std': 'word_count_std'}, inplace=True)
    return stats

def generate_factchecker_stats(dataset_pd_obj, column_grouped_by):
    """Calculate all the statistics above for authors or organisations and save them to a CSV file.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object containing the data.
        column_grouped_by (str): 'author' or 'organisation'.
    """
    stats = pd.DataFrame()
    stats[column_grouped_by] = dataset_pd_obj[column_grouped_by].unique()
    stats = stats.merge(first_to_last_fact_check(dataset_pd_obj, column_grouped_by), on=column_grouped_by)
    stats = stats.merge(number_of_fact_checks(dataset_pd_obj, column_grouped_by), on=column_grouped_by)
    stats = stats.merge(percentage_unique_fact_checks(dataset_pd_obj, column_grouped_by), on=column_grouped_by)
    stats = stats.merge(fact_checking_rate(dataset_pd_obj, column_grouped_by), on=column_grouped_by)
    stats = stats.merge(number_of_fact_checkers(dataset_pd_obj, column_grouped_by), on=column_grouped_by)
    stats = stats.merge(word_count_stats(dataset_pd_obj, column_grouped_by), on=column_grouped_by)
    stats.to_csv("../../../data/" + str(column_grouped_by) + "_stats.csv", index=False)