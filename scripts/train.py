import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline

from constant import MODEL_REGISTRY, NUMERICAL_COLS, CATEGORICAL_COLS

import warnings
warnings.filterwarnings("ignore")


# MLflow Tracking URI
mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]


def load_data(path):
    df = pd.read_csv(path)

    x = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )


def build_preprocessor(
    strategy,
    numerical_cols=NUMERICAL_COLS,
    categorical_cols=CATEGORICAL_COLS
):

    if strategy == "tree":
        encoder = OrdinalEncoder()
        scaler = "passthrough"

    else:  # linear
        encoder = OneHotEncoder()
        scaler = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_cols),
            ("cat", encoder, categorical_cols),
        ]
    )


def train_model(
    experiment_name,
    model_cfg,
    X_train,
    y_train,
    X_test,
    y_test
):

    model_name = model_cfg.get("model_name", None)

    model_instance = MODEL_REGISTRY.get(
        model_name,
        None
    )

    if model_instance is None:
        raise ValueError(
            f"Model {model_name} not found in registry."
        )

    model_hyperparams = model_cfg.get(
        "hyperparameters",
        {}
    )

    preproc_strat = model_cfg.get(
        "preproc_strategy",
        None
    )

    preprocessor = build_preprocessor(
        preproc_strat,
        numerical_cols=NUMERICAL_COLS,
        categorical_cols=CATEGORICAL_COLS
    )

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model_instance(**model_hyperparams)
            )
        ]
    )

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(
        run_name=experiment_name
    ):

        # Log hyperparameters
        mlflow.log_params(
            model_hyperparams
        )

        # Train model
        pipeline.fit(
            X_train,
            y_train
        )

        # Predictions
        y_pred = pipeline.predict(
            X_test
        )

        y_pred_proba = (
            pipeline.predict_proba(X_test)
            if hasattr(
                pipeline,
                "predict_proba"
            )
            else None
        )

        # Metrics
        mlflow.log_metric(
            "accuracy",
            accuracy_score(
                y_test,
                y_pred
            )
        )

        mlflow.log_metric(
            "precision",
            precision_score(
                y_test,
                y_pred
            )
        )

        mlflow.log_metric(
            "recall",
            recall_score(
                y_test,
                y_pred
            )
        )

        mlflow.log_metric(
            "f1",
            f1_score(
                y_test,
                y_pred
            )
        )

        if y_pred_proba is not None:

            mlflow.log_metric(
                "roc_auc",
                roc_auc_score(
                    y_test,
                    y_pred_proba[:, 1]
                )
            )

        # Log model to MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_test[:5],
            registered_model_name=None
        )

    return model_info


def main():

    print("Training started...")

    # Project root directory
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    # Configuration file path
    CONFIG_PATH = os.path.join(
        BASE_DIR,
        "scripts",
        "config.yaml"
    )

    # Dataset path
    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "heart.csv"
    )

    # Load configuration
    with open(
        CONFIG_PATH,
        "r"
    ) as f:

        config = yaml.safe_load(f)

    name_experiment = config.get(
        "name_experiment",
        "Default Experiment"
    )

    # Load dataset
    X_train, X_test, y_train, y_test = load_data(
        DATA_PATH
    )

    print(
        f"mlflow_uri: {mlflow_uri}"
    )

    # Configure MLflow
    mlflow.set_tracking_uri(
        uri=mlflow_uri
    )

    mlflow.set_experiment(
        name_experiment
    )

    # Get models from config.yaml
    models_cfg = config.get(
        "model",
        {}
    )

    # Train all models
    for i, (
        model_name,
        model_cfg
    ) in enumerate(
        models_cfg.items()
    ):

        print(
            f"Training model "
            f"{i + 1}/{len(models_cfg)}: "
            f"{model_name}"
        )

        train_model(
            model_name,
            model_cfg,
            X_train,
            y_train,
            X_test,
            y_test
        )

        print(
            f"Model {model_name} "
            f"trained successfully!"
        )


if __name__ == "__main__":

    print("MLflow version:")

    main()
