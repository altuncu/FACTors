# Codes used for the credibility assessment example in the paper

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

    # Absolute value of bias scores
    org_stats['bias_mean'] = org_stats['bias_mean'].abs()

    # Convert all the values to the rank among all the organisations, except the organisation name
    columns_to_rank = [col for col in org_stats.columns if col != 'organisation']
    org_stats[columns_to_rank] = org_stats[columns_to_rank].rank(method='min', ascending=False)
    
    # Replace the ranks with 1/rank to get the reciprocal of the ranks
    org_stats[columns_to_rank] = 1 / (org_stats[columns_to_rank] * 7.0)

    # Calculate the credibility score by summing up the reciprocal of the ranks
    org_stats['credibility_score'] = -org_stats['bias_mean'] + org_stats['experience_days'] + \
                                      org_stats['number_of_fact_checks'] + org_stats['unique_claim_percentage'] + \
                                      org_stats['fc_frequency_mean'] + org_stats['number_of_authors'] + \
                                      org_stats['word_count_mean']
    org_stats = org_stats.sort_values('credibility_score', ascending=False)
    org_stats['anonymised_org'] = org_stats['organisation'].astype('category').cat.codes + 1
    org_stats['anonymised_org'] = org_stats['anonymised_org'].apply(lambda x: f"o{x}")

    org_stats.to_csv('credibility_scores.csv', index=False)
    
def plot_bar_plot():
    """Plots the bar plot showing the contribution of different factors to the credibility score
    """
    new_column_names = {
        "bias_mean": "Mean political bias",  # negative
        "experience_days": "Fact-checking experience",
        "number_of_fact_checks": "Number of fact-checks",
        "unique_claim_percentage": "Unique claim percentage",
        "fc_frequency_mean": "Mean fact-checking frequency",
        "word_count_mean": "Mean word count",
        "number_of_authors": "Total number of authors"
    }

    # Load and prepare data
    cred_scores = pd.read_csv('credibility_scores.csv')
    cred_scores['organisation'] = cred_scores['organisation'].astype('category').cat.codes + 1
    cred_scores['organisation'] = cred_scores['organisation'].apply(lambda x: f"o{x}")
    grouped = cred_scores.groupby("organisation").mean()

    # Invert negative factor
    grouped['bias_mean'] *= -1

    # Compute final score
    grouped['credibility_score'] = grouped[list(new_column_names.keys())].sum(axis=1)
    grouped.rename(columns=new_column_names, inplace=True)
    
    # Sort by credibility score
    grouped = grouped.sort_values(by='credibility_score', ascending=False)

    factor_cols = [new_column_names[col] for col in new_column_names]
    positive_factors = [col for col in factor_cols if col != "Mean political bias"]
    negative_factors = ["Mean political bias"]
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # Setup figure
    fig, ax = plt.subplots(figsize=(11, 6))

    # Colors
    tab10 = cm.get_cmap("tab10")
    # Manually skip red (index 3) from tab10
    tab10_indices = [i for i in range(10) if i != 3]
    
    bar_width = 0.5

    # Stack positive factors
    bottom = pd.Series(0, index=grouped.index)
    for i, col in enumerate(positive_factors):
        color_index = tab10_indices[i % len(tab10_indices)]
        ax.bar(grouped.index, grouped[col], bottom=bottom, color=tab10(color_index), label=col, width=bar_width)
        bottom += grouped[col]

    # Add negative factor (orange, below zero)
    ax.bar(grouped.index, grouped["Mean political bias"], bottom=0, color='red', label="Mean political bias", width=bar_width)

    # Add black line for credibility score
    ax.plot(grouped.index, grouped['credibility_score'], color='black', linewidth=1.5, marker='.', label="Credibility Score")

    # Styling
    ax.axhline(0, color='grey', linewidth=1)
    ax.set_ylabel("Credibility Score")
    ax.set_xlabel("Fact-checking Organisation")
    plt.xticks(rotation=90)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlim(-0.5, len(grouped.index) - 0.5)
    plt.tight_layout()
    plt.savefig("credibility_scores.pdf", format="pdf", bbox_inches="tight")
    plt.show()
