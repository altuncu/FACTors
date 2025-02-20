# Codes used for the credibility assessment example in the paper

import pandas as pd
import matplotlib.pyplot as plt

def generate_credibility_scores():
    """Assigns credibility scores to the fact-checking organisations based on the factors
    """
    org_stats = pd.read_csv('../statisticalanalysis/organisation_stats.csv')
    biases = pd.read_csv('../politicalbiasdetection/organisation_bias_scores.csv')

    # Merge the two dataframes
    org_stats = org_stats.merge(biases, on='organisation', how='left')
    org_stats = org_stats[['organisation', 'experience_days', 'number_of_fact_checks', 
                           'unique_claim_percentage', 'fc_frequency_mean', 
                           'number_of_authors', 'word_count_mean', 'bias_mean']]

    # Absolute value of bias scores and multiply them with -1 to get the negative effect
    org_stats['bias_mean'] = -org_stats['bias_mean'].abs()

    # Convert all the values to the rank among all the organisations, except the organisation name
    columns_to_rank = [col for col in org_stats.columns if col != 'organisation']
    org_stats[columns_to_rank] = org_stats[columns_to_rank].rank(method='min', ascending=False)
    
    # Replace the ranks with 1/rank to get the reciprocal of the ranks
    org_stats[columns_to_rank] = 1 / org_stats[columns_to_rank]

    # Calculate the credibility score by summing up the reciprocal of the ranks
    org_stats['credibility_score'] = (org_stats['bias_mean'] + org_stats['experience_days'] + \
                                      org_stats['number_of_fact_checks'] + org_stats['unique_claim_percentage'] + \
                                      org_stats['fc_frequency_mean'] + org_stats['number_of_authors'] + \
                                      org_stats['word_count_mean']) / 7.0
    org_stats = org_stats.sort_values('credibility_score', ascending=False)

    org_stats.to_csv('credibility_scores.csv', index=False)
    
def plot_bar_plot():
    """Plots the bar plot showing the contribution of different factors to the credibility score
    """
    new_column_names = {
    "bias_mean": "Mean political bias",
    "experience_days": "Fact-checking experience",
    "number_of_fact_checks": "Number of fact-checks",
    "unique_claim_percentage": "Unique claim percentage",
    "fc_frequency_mean": "Mean fact-checking frequency",
    "word_count_mean": "Mean word count",
    "number_of_authors": "Total number of authors"
    }   
    
    cred_scores = pd.read_csv('credibility_scores.csv')
    cred_scores = cred_scores.sort_values('credibility_score', ascending=False)
    cred_scores['organisation'] = cred_scores['organisation'].astype('category').cat.codes + 1
    cred_scores['organisation'] = cred_scores['organisation'].apply(lambda x: f"o{x}")
    factors = cred_scores.columns.difference(["organisation", "credibility_score"])  

    contributions = cred_scores.groupby("organisation")[factors].sum() / factors.size
    cred_scores_grouped = cred_scores.groupby("organisation")['credibility_score'].mean()
    contributions['credibility_score'] = cred_scores_grouped
    contributions = contributions.sort_values(by='credibility_score', ascending=False).drop('credibility_score', axis=1)
    contributions.rename(columns=new_column_names, inplace=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    contributions.plot(kind="bar", stacked=True, colormap="tab10", ax=ax)

    ax.set_ylabel("Credibility Score")
    ax.set_xlabel("Fact-checking Organisation")
    plt.xticks(rotation=90)
    ax.legend(title="Factors")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    #plt.savefig("credibilityscores.pdf", format="pdf", bbox_inches="tight")
    #plt.show()
