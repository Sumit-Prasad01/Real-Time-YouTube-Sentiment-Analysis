import os 
import mlflow
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI")

df = pd.read_csv("https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")

print(df.head())

df.dropna(inplace = True)
df.drop_duplicates(inplace = True)
df = df[~(df['clean_comment'].str.strip() == '')]

import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_comment(comment):

    comment = comment.lower()
    comment = comment.strip()
    comment = re.sub(r'\n', ' ',comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    lemmatize = WordNetLemmatizer()
    comment = ' '.join([lemmatize.lemmatize(word) for word in comment.split()])

    return comment

df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)

print(df.head())

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

vectorizer = CountVectorizer(max_features = 10000)

X = vectorizer.fit_transform(df['clean_comment']).toarray()
y = df['category']

print(X)
print(X.shape)

print(y)
print(y.shape)

mlflow.set_tracking_uri(MLFLOW_URI)

mlflow.set_experiment('RF Baseline')

X_train, X_test, y_tarin, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with mlflow.start_run():

    mlflow.set_tag("mlflow.runName", "RandomForest_Baseline_TrainTestSplit")
    mlflow.set_tag("experiment_type", "baseline")
    mlflow.set_tag("model_type", "RandomForestClassifier")

    mlflow.set_tag("mlflow.runName", "Baseline RandomForest model for sentiment snalysis using Bag of words (BoW) with a simple tarin-test-split")

    mlflow.log_param("vectorizer_type", "CountVectorizer")
    mlflow.log_param("vectorizer_max_features", vectorizer.max_features)

    n_estimators = 200
    max_depth = 15

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_tarin)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    classification_rep = classification_report(y_test, y_pred, output_dict = True)

    for label, metrices in classification_rep.items():
        if isinstance(metrices, dict):
            for metric, value in metrices.items():
                mlflow.log_metric(f"{label}_{metric}", value)

    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("../visuals/Baseline_Model/confusion_matrix.png")
    mlflow.log_artifact("/content/confusion_matrix.png")

    mlflow.sklearn.log_model(model, 'random_forest_model')

    df.to_csv('dataset.csv', index = False)
    mlflow.log_artifact('/content/dataset.csv')

print(f"Accuracy : ",{accuracy})
print(classification_report(y_test, y_pred))

pd.read_csv('reddit_preprocessing.csv', index = False)
new_df = pd.read_csv('reddit_preprocessing')

print(new_df.head())
