# Functions used to normalise the verdicts of the FACTors dataset. 

from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from itertools import islice
import pandas as pd
import string
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import re

# Mapping of original verdicts to normalised verdicts
verdict_mapping = { # Covers 72,309 fact-checks of 33 organisations
    "false": ['satire', 'pants-fire', 'legend', 'lie', 'no truth', 'false', 'hoax', 'fake', 'incorrect', 'totally false', 'inaccurate', 'fake news', 'fabricated', 'deepfake', 'untrue', 'wrong'],
    "true": ['fact', 'legitimate', 'true', 'correct', 'accurate'],
    "misleading": ['clickbait', 'exaggeration', 'misattributed', 'outdated', 'miscaptioned', 'out of context', 'incorrect attribution', 'false attribution', 'misleading', 'no longer true', 'misrepresentation', 'misinterpretation'],
    "partially true": ['multiple', 'mixture', 'barely-true', 'mostly-true', 'mostly accurate', 'mostly correct', 'mostly true', 'mostly inaccurate', 'mostly false', 'partly false', 'partially false', 'partly true', 'partly correct', 'partially true', 'half true', 'half-true', 'partially correct', 'halftrue', 'partly true and false'],
    "unverifiable": ['uncheckable', 'no basis', 'more context needed', 'inconclusive', 'cannot be proven', 'insufficient details', 'unfounded', 'unproven', 'no evidence', 'unsubstantiated', 'insufficient evidence', 'no sufficient evidence', 'without evidence', 'lacks evidence', 'evidence is lacking'],
    "other": ['no rating found']
}

def normalised_verdict(text):
    """Helper function to map original verdicts to normalised verdicts.

    Args:
        text (str): Original verdict

    Returns:
        str: Normalised verdict
    """
    for label, keywords in verdict_mapping.items():
        for keyword in keywords:
            if keyword == text:
                return label
    return None

def assign_normalised_verdicts(dataset_pd_obj):
    """Assigns normalised verdicts by mapping original verdicts to normalised verdicts.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataframe containing the dataset

    Returns:
        pd.DataFrame: Dataframe with mapped normalised verdicts
    """
    dataset_pd_obj['normalised_rating'] = dataset_pd_obj['original_verdict'].apply(normalised_verdict)
    return dataset_pd_obj

def preprocess_text(text):
    """Helper function for preprocessing text.

    Args:
        text (string): Input text to be preprocessed

    Returns:
        string: Preprocessed text
    """
    text = text.strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?\'\"\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

# Initialise the model and tokeniser
model_name = "roberta-base"
tokeniser = AutoTokenizer.from_pretrained(model_name)

# Mapping of labels to their corresponding strings
label_mapping = {0: "false", 1: "misleading", 2: "other", 3: "partially true", 4: "true", 5: "unverifiable"}

def tokenise_function(examples):
    # Tokenise the text
    examples["text"] = [preprocess_text(text) for text in examples["text"]]
    tokenised = tokeniser(examples["text"], padding=True, truncation=True, max_length=512)
    if "label" in examples:
        tokenised["labels"] = examples["label"]
    return tokenised

def compute_metrics(eval_pred):
    """Compute the accuracy and F1 score of the model.

    Args:
        eval_pred: Predictions of the model.

    Returns:
        dict: Scores of the model.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1_score": f1}

def finetune_and_predict(threshold=0.5):
    """Fine-tunes the model on the FACTors dataset and predicts the normalised verdicts
    of the unlabelled rows.

    Args:
        threshold (float, optional): Confidence level threshold. Defaults to 0.5.
    """
    dataset = pd.read_csv("data/FACTors.csv")
    label_encoder = LabelEncoder()
    dataset.loc[dataset['normalised_rating'].notnull(), 'normalised_rating'] = label_encoder.fit_transform(dataset.loc[dataset['normalised_rating'].notnull(), 'normalised_rating'])
    labeled_dataset = Dataset.from_pandas(dataset[dataset['normalised_rating'].notnull()].rename(columns={"content": "text", "normalised_rating": "label"})[['text', 'label']])
    unlabeled_dataset = Dataset.from_pandas(dataset[dataset['normalised_rating'].isnull()].rename(columns={"content": "text"})[['text']])

    tokenised_labeled = labeled_dataset.map(tokenise_function, batched=True)
    tokenised_labeled = tokenised_labeled.train_test_split(test_size=0.1)
    tokenised_unlabeled = unlabeled_dataset.map(tokenise_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained('./model/checkpoint-12204')

    class_weights = compute_class_weight('balanced', classes=np.unique(dataset['normalised_rating'].dropna()), y=dataset['normalised_rating'].dropna())
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    #loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    training_args = TrainingArguments(output_dir="./results", eval_strategy="epoch",
                                      save_strategy="epoch", num_train_epochs=3, learning_rate=3e-5,
                                      per_device_train_batch_size=16, per_device_eval_batch_size=16,
                                      weight_decay=0.01, logging_dir="./logs", logging_steps=10,
                                      load_best_model_at_end=True, metric_for_best_model="accuracy"
                                     )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenised_labeled["train"],
        eval_dataset=tokenised_labeled["test"], processing_class=tokeniser,
        compute_metrics=compute_metrics
    )

    trainer.train()
    #trainer = Trainer(model=model, tokenizer=tokeniser)
    test_results = trainer.evaluate()
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['eval_f1_score']:.4f}")

    predictions = trainer.predict(tokenised_unlabeled)
    normalised_verdicts = np.argmax(predictions.predictions, axis=1)

    probs = F.softmax(torch.tensor(predictions.predictions), dim=1)
    max_probs = torch.max(probs, dim=1).values.numpy()

    high_confidence_mask = max_probs >= threshold
    high_confidence_df = unlabeled_dataset[high_confidence_mask].to_pandas()
    high_confidence_df["normalised_verdict"] = normalised_verdicts[high_confidence_mask]

    high_confidence_df.to_csv("normalised_verdicts.csv", index=False)

    print(f"{high_confidence_df.shape[0]} rows have been labelled with high confidence.")
    
def export_unlabelled_rows(dataset_pd_obj):
    """Export the unlabelled rows to a CSV file for manual labelling.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataframe containing the dataset

    Returns:
        pd.DataFrame: Dataframe with unlabelled rows exported to a CSV file
    """
    dataset_pd_obj['content_clean'] = dataset_pd_obj['content'].apply(preprocess_text)
    predictions = pd.read_csv("normalised_verdicts_lowconf.csv")
    predictions['content_clean'] = predictions['content'].apply(preprocess_text)
    merged_df = dataset_pd_obj.merge(predictions, on="content_clean", how="left", suffixes=("_df1", "_df2"))
    merged_df["normalised_rating"] = merged_df["normalised_rating_df1"].combine_first(merged_df["normalised_rating_df2"])
    merged_df = merged_df[["row_id", "original_verdict", "normalised_rating"]]
    print("Number of remaining unlabelled rows: ", merged_df['normalised_rating'].isnull().sum())
    merged_df[merged_df['normalised_rating'].isnull()].to_csv("data/FACTors_unlabelled.csv", index=False)
    return dataset_pd_obj

def assign_predicted_verdicts(dataset_pd_obj):
    """Assigns the predicted verdicts to the dataset.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object

    Returns:
        pd.DataFrame: Dataset object with predicted verdicts
    """
    predictions = pd.read_csv("normalised_verdicts_lowconf.csv")
    predictions = predictions[["content", "normalised_rating"]]
    predictions['normalised_rating'] = predictions['normalised_rating'].map(label_mapping)
    dataset_pd_obj = dataset_pd_obj.merge(predictions, on="content", how="left", suffixes=("_df1", "_df2"))
    dataset_pd_obj['normalised_rating'] = dataset_pd_obj['normalised_rating_df1'].combine_first(dataset_pd_obj['normalised_rating_df2'])
    dataset_pd_obj = dataset_pd_obj.drop(columns=["normalised_rating_df1", "normalised_rating_df2"])
    return dataset_pd_obj

def assign_annotated_verdicts(dataset_pd_obj):
    """Assigns the manually annotated verdicts to the dataset.

    Args:
        dataset_pd_obj (pd.DataFrame): Dataset object

    Returns:
        pd.DataFrame: Dataset object with annotated verdicts
    """
    annotations = pd.read_csv("data/FACTors_unlabelled.csv")
    # Merge the annotations with the dataset for the same row_id
    dataset_pd_obj = dataset_pd_obj.merge(annotations, on="row_id", how="left", suffixes=("_df1", "_df2"))
    # Combine the normalised_rating columns
    dataset_pd_obj['normalised_rating'] = dataset_pd_obj['normalised_rating_df1'].combine_first(dataset_pd_obj['normalised_rating_df2'])
    # Drop the redundant columns
    dataset_pd_obj = dataset_pd_obj.drop(columns=["normalised_rating_df1", "normalised_rating_df2"])
    return dataset_pd_obj