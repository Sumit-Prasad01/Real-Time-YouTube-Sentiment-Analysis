import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI ")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Exp 3 - Tfidf Trigram max_features")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
     

df = pd.read_csv("/content/reddit_preprocessing.csv").dropna(subset = ['clean_comment'])
print(df.shape)

def run_experiment_tdidf_max_features(max_features):

    ngram_range = (1,3)

    vectorizer = TfidfVectorizer(ngram_range, ngram_range, max_features = max_features)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    with mlflow.set_tags("mlflow.runName", f"RandomForest with TF_IDF Trigrams, max_features = {max_features}"):

        mlflow.set_tag("mlflow.runName", f"TFIDF_Trigrams_max_features_{max_features}")
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        mlflow.set_tag("description", f"RandomForest with TF-IDF Trigrams, max_features={max_features}")

        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("vectorizer_max_features", max_features)

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
        for label, metrices in classification_rep.items():
            if isinstance(metrices, dict):
                for metric, value in metrices.items():
                    mlflow.log_metric(f"{metric}-{label}", value)


        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: TF-IDF Trigrams, max_features={max_features}")
        plt.savefig("../viusals/Tfidf(1,3)/confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()


        mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_trigrams_{max_features}")


max_features_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

for max_features in max_features_values:
    run_experiment_tdidf_max_features(max_features)