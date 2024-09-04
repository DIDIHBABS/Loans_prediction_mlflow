import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv "
    try:
        df = pd.read_csv(URL, sep=';')
        return df
    except Exception as e:
        raise e


def evaluate_fun(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, mae, r2


def main(alpha, l1_ratio):
    df = load_data()
    TARGET = 'quality'
    X = df.drop(columns=TARGET, axis=1)
    y = df[TARGET]
    X_train,  X_test, y_train, y_test = train_test_split(X,y, random_state=6, train_size=0.2)
    mlflow.set_experiment("Ml-Model-1")
    with mlflow.start_run():
        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=6)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse, mae, r2 = evaluate_fun(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "trained_model",
                                 signature=signature,
                                 input_example= input_example) # model, folder name


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--alpha', '-a', type=float, default=0.2)
    args.add_argument('--l1_ratio', '-l1', type=float, default=0.3)
    parsed_args = args.parse_args()
    # parsed_args.param1
    main(parsed_args.alpha, parsed_args.l1_ratio)