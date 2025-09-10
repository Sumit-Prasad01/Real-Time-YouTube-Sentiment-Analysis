import os 
import mlflow
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Exp 2 - BoW vs Tfidf")


import pandas as pd
import seaborn as sns
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/content/reddit_preprocessing.csv").dropna(subset = ['clean_comment'])
df.shape

def run_experiment(vectorizer_type, ngram_range, vectorizer_max_features, vectorizer_name):

    if vectorizer_type == "BoW":
        vectorizer = CountVectorizer(ngram_range = ngram_range, max_features = vectorizer_max_features)
    else :
        vectorizer = TfidfVectorizer(ngram_range = ngram_range, max_features = vectorizer_max_features)

    X_tarin, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size = 0.2, random_state = 42, stratify = df['category'])

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", f"{vectorizer_name}_{ngram_range}_RandomForest")
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        mlflow.set_tag("description", f"RandomForest with {vectorizer_name}, ngram_range={ngram_range}, max_features={vectorizer_max_features}")

        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("vectorizer_max_features", vectorizer_max_features)

        n_estimators = 200
        max_depth = 15

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = 42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        classification_rep = classification_report(y_test, y_pred, output_dict = True)
        for  label, metrices in classification_rep.items():
            if isinstance(metrices, dict):
                for metric, value in metrices.items():
                    mlflow.log_metric(f"{label}_{metric}", value)


        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize = (8,6))
        sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix : {vectorizer_name}, {ngram_range} ")
        plt.savefig("../visuals/BoW_vs_Tfidf/confusion_matrix.png")
        plt.close()

        mlflow.sklearn.log_model(model, f"random_forest_model_{vectorizer_name}_{ngram_range}")


ngram_ranges = [(1,1), (1,2), (1,3)]
max_features = 5000

for ngram_range in ngram_ranges:

    run_experiment("BoW", ngram_range, max_features, vectorizer_name= 'BoW')
    run_experiment("TF-IDF", ngram_range, max_features, vectorizer_name= 'TF-IDF')