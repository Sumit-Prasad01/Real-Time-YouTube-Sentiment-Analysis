import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv('MLFLOW_URI')

mlflow.set_tracking_uri("MLFLOW_URI")
mlflow.set_experiment('Exp 4 - Handling Imbalanced Data')

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
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


def run_imbalanced_experiment(imbalance_method):
    ngram_range = (1,3)
    max_features = 10000

    X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

    vectorizer = TfidfVectorizer(ngram_range = ngram_range, max_features = max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.fit_transform(X_test)

    if imbalance_method == 'class_weights':
        class_weight = 'balanced'

    else:
        class_weight = None

        if imbalance_method == 'oversampling':
            smote = SMOTE(random_state = 42)
            X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

        elif imbalance_method == 'adasyn':
            adasyn = ADASYN(random_state=42)
            X_train_vec, y_train = adasyn.fit_resample(X_train_vec, y_train)

        elif imbalance_method == 'undersampling':
            rus = RandomUnderSampler(random_state=42)
            X_train_vec, y_train = rus.fit_resample(X_train_vec, y_train)

        elif imbalance_method == 'smote_enn':
            smote_enn = SMOTEENN(random_state=42)
            X_train_vec, y_train = smote_enn.fit_resample(X_train_vec, y_train)
    
    with mlflow.start_run() as run:

        mlflow.set_tag('mlflow.runName', f"Imbalance_{imbalance_method}_RandomForest_TDIDF_Trigrams")
        mlflow.set_tag("experiment_type", "imbalance_handling")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        mlflow.set_tag("description", f"RandomForest with TF-IDF Trigrams, imbalance hadling method = {imbalance_method}")

        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param('ngram_range', ngram_range)
        mlflow.log_param("vectorizer_max_features", max_features)

        n_estimators = 200
        max_depth = 15

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("vectorizer_max_features", max_features)


        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = 42, class_weight = class_weight)
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        classification_rep = classification_report(y_test, y_pred, output_dict = True)
        for label, metrices in classification_rep.items():
            if isinstance(metrices, dict):
                for metric, value in metrices.items():
                    mlflow.log_metric(f"{label}-{metric}", value)

        
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: TF-IDF Trigrams, Imbalance={imbalance_method}")
        confusion_matrix_filename = f"confusion_matrix_{imbalance_method}.png"
        plt.savefig(f'../visuals/Imbalanced_Data/{confusion_matrix_filename}')
        mlflow.log_artifact(confusion_matrix_filename)
        plt.close()


        mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_trigrams_imbalance_{imbalance_method}")


imbalance_methods = ['class_weights', 'oversampling', 'adasyn', 'undersampling', 'smote_enn']

for method in imbalance_methods:
    run_imbalanced_experiment(method)