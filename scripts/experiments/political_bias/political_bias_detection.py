# Codes for the political bias detection example in the paper

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

# Initialise the bias detection model
dataset = pd.read_csv("../../../data/FACTors.csv")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")

# Get unique contents since a content can include multiple claims, corresponding to multiple rows in the dataset
unique_contents = dataset[['article_id', 'author', 'organisation', 'content']].drop_duplicates(subset=['content'])

def get_bias_score(text):
    """Obtain the probabilities of the political bias of the given text

    Args:
        text (str): Input text

    Returns:
        list: List of probabilities of the political bias, [left, center, right]
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = logits.softmax(dim=-1)[0].tolist()
    return probabilities

# [0] -> left 
# [1] -> center
# [2] -> right
def save_bias_scores():
    """Save the predicted bias scores for each content in the dataset
    """
    unique_contents['bias_scores'] = unique_contents['content'].apply(get_bias_score)
    unique_contents[['article_id', 'author', 'organisation', 'bias_scores']].to_csv("data/content_bias_scores.csv", index=False)

def generate_orgs_plot():
    """Based on the bias scores, generate a plot showing the average political bias of each fact-checking organisation
    """
    dataset = pd.read_csv("content_bias_scores.csv")
    bias_map = {0: -1, 1: 0, 2: 1}
    bias_labels = {-1: "Left", 0: "Centre", 1: "Right"}

    dataset["bias_scores"] = dataset["bias_scores"].apply(ast.literal_eval)
    dataset["max_bias_index"] = dataset["bias_scores"].apply(lambda x: np.argmax(x))
    dataset["bias_numeric"] = dataset["max_bias_index"].map(bias_map)
    organisation_stats = dataset.groupby("organisation")["bias_numeric"].agg(["mean", "std"]).reset_index()
    organisation_stats["bias_category"] = organisation_stats["mean"].round().map(bias_labels)
    organisation_stats = organisation_stats.sort_values(by=['mean'], ascending=False)
    organisation_stats.to_csv("./org_bias_scores.csv", index=False)

    plt.figure(figsize=(7, 10))
    plt.errorbar(organisation_stats["mean"], organisation_stats["organisation"], xerr=organisation_stats["std"],
                 fmt="o", color="darkblue", ecolor="royalblue", capsize=5, alpha=0.7)

    plt.axvline(0, color="gray", linestyle="--")
    plt.xticks(ticks=[-1, 0, 1], labels=["Left", "Centre", "Right"])
    plt.xlabel("Political Bias")
    plt.ylabel("Fact-checking Organisation")
    plt.savefig("politicalbias.pdf", format="pdf", bbox_inches="tight")
    plt.show()