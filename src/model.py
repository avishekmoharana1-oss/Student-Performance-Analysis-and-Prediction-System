
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_and_evaluate_model(data, result_dir, chart_dir):
    """
    Train, tune and evaluate a Linear Regression model
    for predicting students' final grades (G3).
    """

    result_dir = Path(result_dir)
    chart_dir = Path(chart_dir)

    result_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate dataset.
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The dataset is empty.")

    if "G3" not in data.columns:
        raise ValueError("The dataset must contain G3.")

    # 2. Separate features and target.
    X = data.drop(columns=["G3"]).copy()
    y = pd.to_numeric(data["G3"], errors="coerce")

    valid_rows = y.notna() & np.isfinite(y)

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    if len(X) < 10:
        raise ValueError(
            "At least 10 valid student records are required."
        )

    # Remove unsupported feature columns.
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    supported_columns = numeric_columns + categorical_columns

    X = X[supported_columns]

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns were found.")

    # 3. Create preprocessing pipelines.
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True
                    )
                )
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # 4. Build the complete machine-learning pipeline.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regression", LinearRegression())
        ]
    )

    # 5. Split data before training and tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    if len(X_train) < 5:
        raise ValueError(
            "Not enough training records for cross-validation."
        )

    folds = min(5, len(X_train))

    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    # 6. Hyperparameter tuning.
    parameter_grid = {
        "regression__fit_intercept": [True, False]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    print("\nStarting Linear Regression hyperparameter tuning...")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest model parameters:")
    print(grid_search.best_params_)

    # Save tuning results.
    tuning_results = pd.DataFrame(
        grid_search.cv_results_
    )

    tuning_results.to_csv(
        result_dir / "hyperparameter_tuning_results.csv",
        index=False
    )

    # 7. Evaluate on unseen test data.
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # 8. Cross-validation using training data only.
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2"
    )

    best_cv_rmse = np.sqrt(
        -grid_search.best_score_
    )

    # 9. Save actual and predicted grades.
    comparison = pd.DataFrame(
        {
            "Actual_Final_Grade": y_test.to_numpy(),
            "Predicted_Final_Grade": predictions,
            "Prediction_Error": (
                y_test.to_numpy() - predictions
            )
        }
    )

    comparison.to_csv(
        result_dir / "actual_vs_predicted.csv",
        index=False
    )

    # 10. Save evaluation metrics.
    metrics = {
        "Mean Absolute Error (MAE)": float(mae),
        "Root Mean Squared Error (RMSE)": float(rmse),
        "R-squared (R2)": float(r2),
        "Cross-validation mean R2": float(cv_scores.mean()),
        "Cross-validation standard deviation": float(cv_scores.std()),
        "Best tuning CV RMSE": float(best_cv_rmse),
        "Best fit_intercept setting": str(
            grid_search.best_params_["regression__fit_intercept"]
        ),
        "Training records": int(len(X_train)),
        "Testing records": int(len(X_test)),
        "Total records used": int(len(X))
    }

    metrics_df = pd.DataFrame(
        [
            {"Metric": name, "Value": value}
            for name, value in metrics.items()
        ]
    )

    metrics_df.to_csv(
        result_dir / "evaluation_metrics.csv",
        index=False
    )

    # 11. Create actual-versus-predicted chart.
    actual_values = y_test.to_numpy()

    lower = min(
        float(actual_values.min()),
        float(predictions.min())
    )

    upper = max(
        float(actual_values.max()),
        float(predictions.max())
    )

    plt.figure(figsize=(7, 6))

    plt.scatter(
        actual_values,
        predictions,
        alpha=0.75
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.title("Actual vs Predicted Final Grades")
    plt.xlabel("Actual Final Grade (G3)")
    plt.ylabel("Predicted Final Grade")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        chart_dir / "actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nModel training and evaluation completed.")

    return metrics
