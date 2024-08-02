import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn

if __name__ == "__main__":

    #dataset

    iris = load_iris()

    X, y = iris.data, iris.target


    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a classifier (Random Forest in this case)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    score = clf.score(X, y)

    print("Score: %s" % score)

    mlflow.log_metric("score", score)

    mlflow.sklearn.log_model(clf, "model")

    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
