# Functions for finding overlapping claims in the dataset and assigning unique claim ids to each group of overlapping claims.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining
import networkx as nx

model = SentenceTransformer("all-MiniLM-L6-v2")

def find_similar_claims_sampling(dataset_pd_obj, sampling=False):
    """Find similar claim pairs in the dataset using the SentenceTransformer model.
    This function is used for identifying the optimum cosine similarity threshold for the model.

    Args:
        dataset_pd_obj (pd.DataFrame): The dataset to find similar claims in.
        sampling (bool, optional): If True, get 1000 random samples. Defaults to False.
    """
    claims = dataset_pd_obj['claim'].to_list()
    lookup_table_id = dict(zip(dataset_pd_obj['claim'], dataset_pd_obj['row_id']))
    lookup_table_org = dict(zip(dataset_pd_obj['claim'], dataset_pd_obj['organisation']))
    similarity_scores = paraphrase_mining(model, claims, show_progress_bar=True, top_k=100)
    similarity_scores = [item for item in similarity_scores 
                         if item[0] > 0.75 and 
                         lookup_table_org.get(claims[item[1]]) != lookup_table_org.get(claims[item[2]])]
    similarity_scores = [[lookup_table_id.get(claims[c[1]]), claims[c[1]], lookup_table_id.get(claims[c[2]]), claims[c[2]], c[0]] for c in similarity_scores]
    similar_claims = pd.DataFrame(columns=['id1', 'claim1', 'id2', 'claim2', 'score'],
                                  data=similarity_scores).sort_values(by='score', ascending=False)
    similar_claims.to_csv('./similar_claims.csv', index=False)
    if sampling:
        similarity_scores = sorted(similarity_scores, key=lambda x: x[4], reverse=True)
        sample_indices = np.random.randint(0, len(similarity_scores), 1000)
        samples = pd.DataFrame(columns=['id1', 'claim1', 'id2', 'claim2', 'score'],
                                 data=[similarity_scores[i] for i in sample_indices])
        samples.to_csv('./similar_claims_sample.csv', index=False)
        
def find_overlapping_claims(dataset_pd_obj):
    """Find overlapping claim pairs based on the obtained cosine similarity threshold.

    Args:
        dataset_pd_obj (pd.DataFrame): The dataset to find similar claims in.
    """
    claims = dataset_pd_obj['claim'].to_list()
    lookup_table_id = dict(zip(dataset_pd_obj['claim'], dataset_pd_obj['row_id']))
    lookup_table_org = dict(zip(dataset_pd_obj['claim'], dataset_pd_obj['organisation']))
    similarity_scores = paraphrase_mining(model, claims, show_progress_bar=True, top_k=100)
    similarity_scores = [item for item in similarity_scores 
                         if item[0] > 0.88 and 
                         lookup_table_org.get(claims[item[1]]) != lookup_table_org.get(claims[item[2]])]
    similarity_scores = [[lookup_table_id.get(claims[c[1]]), lookup_table_id.get(claims[c[2]]), c[0]] for c in similarity_scores]
    similar_claims = pd.DataFrame(columns=['id1', 'id2', 'score'],
                                  data=similarity_scores)
    similar_claims.to_csv('./overlapping_claims.csv', index=False)

def plot_precision_threshold(similar_claims_path='similar_claims_sample.csv'):
    """Plot the precision vs. threshold graph for the similar claims.

    Args:
        similar_claims_path (str, optional): Annotated samples of similar claims. Defaults to 'similar_claims_sample.csv'.
    """
    df = pd.read_csv(similar_claims_path)

    y_scores = df["score"].values
    y_true = df["similar"].values

    precision, _, thresholds_pr = precision_recall_curve(y_true, y_scores)
    desired_precision = 0.95
    valid_indices = np.where(precision >= desired_precision)[0]

    if len(valid_indices) > 0:
        best_idx = valid_indices[0]
        best_threshold = thresholds_pr[best_idx]
        best_precision = precision[best_idx]
    else:
        best_threshold = thresholds_pr[-1]
        best_precision = precision[-1]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds_pr, precision[:-1], color="blue", label="Precision")
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"Optimal Threshold = {best_threshold:.2f}")
    plt.scatter(best_threshold, best_precision, color="red", s=100, label=f"Precision = {best_precision:.2f}")

    plt.xlabel("Threshold (Cosine Similarity)")
    plt.ylabel("Precision")
    plt.title("Precision vs. Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def match_similar_claims(dataset_pd_obj):
    """Combine overlapping claim pairs into groups.

    Args:
        dataset_pd_obj (pd.DataFrame): The dataset to find similar claims in.

    Returns:
        pd.DataFrame: The dataset with the claim ids given to each claim group.
    """
    df = pd.read_csv("similar_claims.csv")

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['id1'], row['id2'])

    claim_groups = {claim: i for i, group in enumerate(nx.connected_components(G)) for claim in group}
    dataset_pd_obj['claim_id'] = dataset_pd_obj['row_id'].map(claim_groups).fillna(-1).astype(int)
    unique_id_start = max(claim_groups.values(), default=0) + 1
    dataset_pd_obj.loc[dataset_pd_obj['claim_id'] == -1, 'claim_id'] = range(unique_id_start, unique_id_start + dataset_pd_obj['claim_id'].eq(-1).sum())
    
    return dataset_pd_obj
