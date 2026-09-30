import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from constant import (
    MODEL_REGISTRY,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "scripts",
    "config.yaml"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "heart.csv"
)


# --------------------------------------------------
# LOAD CONFIGURATION
# --------------------------------------------------

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data(path):

    print(f"Loading dataset from: {path}")

    df = pd.read_csv(path)

    print(f"Dataset shape: {df.shape}")

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# --------------------------------------------------
# CREATE PREPROCESSOR
# --------------------------------------------------

def create_preprocessor(strategy):

    if strategy == "tree":

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    "passthrough",
                    NUMERICAL_COLS
                ),
                (
                    "cat",
                    "passthrough",
                    CATEGORICAL_COLS
                )
            ]
        )

    else:

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    NUMERICAL_COLS
                ),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    CATEGORICAL_COLS
                )
            ]
        )

    return preprocessor


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("MLflow version:", mlflow.__version__)
    print("Training started...")

    mlflow_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )

    print(
        "MLflow URI:",
        mlflow_uri
    )

    print(
        "Config path:",
        CONFIG_PATH
    )

    print(
        "Data path:",
        DATA_PATH
    )

    # --------------------------------------------------
    # MLflow Configuration
    # --------------------------------------------------

    mlflow.set_tracking_uri(
        mlflow_uri
    )

    experiment_name = config[
        "name_experiment"
    ]

    mlflow.set_experiment(
        experiment_name
    )

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = (
        load_data(DATA_PATH)
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # --------------------------------------------------
    # Get Models From config.yaml
    # --------------------------------------------------

    models_config = config["model"]

    total_models = len(
        models_config
    )

    # --------------------------------------------------
    # Train Each Model
    # --------------------------------------------------

    for index, (
        model_version,
        model_config
    ) in enumerate(
        models_config.items(),
        start=1
    ):

        model_name = model_config[
            "model_name"
        ]

        preprocessing_strategy = (
            model_config[
                "preproc_strategy"
            ]
        )

        hyperparameters = model_config.get(
            "hyperparameters",
            {}
        )

        print(
            f"\nTraining model "
            f"{index}/{total_models}: "
            f"{model_version}"
        )

        print(
            "Model:",
            model_name
        )

        print(
            "Preprocessing:",
            preprocessing_strategy
        )

        # --------------------------------------------------
        # Get Model Class
        # --------------------------------------------------

        model_class = MODEL_REGISTRY[
            model_name
        ]

        # --------------------------------------------------
        # Create Model
        # --------------------------------------------------

        model = model_class(
            **hyperparameters
        )

        # --------------------------------------------------
        # Create Preprocessor
        # --------------------------------------------------

        preprocessor = (
            create_preprocessor(
                preprocessing_strategy
            )
        )

        # --------------------------------------------------
        # Create Pipeline
        # --------------------------------------------------

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        # --------------------------------------------------
        # MLflow Run
        # --------------------------------------------------

        with mlflow.start_run(
            run_name=model_version
        ):

            # Train model
            pipeline.fit(
                X_train,
                y_train
            )

            # Prediction
            y_pred = pipeline.predict(
                X_test
            )

            # --------------------------------------------------
            # Calculate Metrics
            # --------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            # --------------------------------------------------
            # ROC-AUC
            # --------------------------------------------------

            try:

                if hasattr(
                    pipeline,
                    "predict_proba"
                ):

                    y_probability = (
                        pipeline.predict_proba(
                            X_test
                        )[:, 1]
                    )

                elif hasattr(
                    pipeline,
                    "decision_function"
                ):

                    y_probability = (
                        pipeline.decision_function(
                            X_test
                        )
                    )

                else:

                    y_probability = None

                if y_probability is not None:

                    roc_auc = roc_auc_score(
                        y_test,
                        y_probability
                    )

                else:

                    roc_auc = 0.0

            except Exception as error:

                print(
                    "ROC-AUC calculation failed:",
                    error
                )

                roc_auc = 0.0

            # --------------------------------------------------
            # Log Parameters
            # --------------------------------------------------

            mlflow.log_param(
                "model_version",
                model_version
            )

            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "preprocessing_strategy",
                preprocessing_strategy
            )

            if hyperparameters:

                mlflow.log_params(
                    hyperparameters
                )

            # --------------------------------------------------
            # Log Metrics
            # --------------------------------------------------

            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "precision",
                precision
            )

            mlflow.log_metric(
                "recall",
                recall
            )

            mlflow.log_metric(
                "f1",
                f1
            )

            mlflow.log_metric(
                "roc_auc",
                roc_auc
            )

            # --------------------------------------------------
            # Log Model
            # --------------------------------------------------

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model"
            )

            # --------------------------------------------------
            # Print Results
            # --------------------------------------------------

            print(
                f"{model_version} completed"
            )

            print(
                f"Accuracy : {accuracy:.4f}"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall   : {recall:.4f}"
            )

            print(
                f"F1 Score : {f1:.4f}"
            )

            print(
                f"ROC-AUC  : {roc_auc:.4f}"
            )

    print(
        "\n========================================"
    )

    print(
        "Training completed successfully."
    )

    print(
        "========================================"
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()



