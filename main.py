
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import train_and_evaluate_model


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "student_performance_dataset.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "visualizations"
RESULT_DIR = OUTPUT_DIR / "evaluation_results"


# --------------------------------------------------
# CREATE OUTPUT FOLDERS
# --------------------------------------------------

def create_output_folders():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE, sep=";")

    required_columns = [
        "G1", "G2", "G3", "studytime", "absences"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The dataset is empty.")

    return data


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def clean_dataset(data):
    data = data.copy()

    original_count = len(data)

    data = data.drop_duplicates()

    data["G3"] = pd.to_numeric(
        data["G3"], errors="coerce"
    )

    data["G3"] = data["G3"].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(subset=["G3"])

    data = data[data["G3"].between(0, 20)]

    if data.empty:
        raise ValueError(
            "No valid student records remain after cleaning."
        )

    data = data.reset_index(drop=True)

    removed_count = original_count - len(data)

    print("\nDATA CLEANING SUMMARY")
    print("-" * 45)
    print(f"Original records: {original_count}")
    print(f"Remaining records: {len(data)}")
    print(f"Records removed: {removed_count}")

    return data


# --------------------------------------------------
# MISSING VALUE ANALYSIS
# --------------------------------------------------

def analyze_missing_values(data):

    missing_report = pd.DataFrame({
        "Missing_Count": data.isnull().sum(),
        "Missing_Percentage": data.isnull().mean() * 100
    })

    missing_report.to_csv(
        RESULT_DIR / "missing_values_report.csv"
    )

    print("\nMISSING VALUE REPORT")
    print("-" * 45)
    print(missing_report)

    return missing_report


# --------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

def perform_eda(data):

    print("\n" + "=" * 55)
    print("STUDENT PERFORMANCE ANALYSIS")
    print("=" * 55)

    print("\nDataset shape:", data.shape)

    print("\nFirst five records:")
    print(data.head())

    print("\nDataset columns:")
    print(data.columns.tolist())

    analyze_missing_values(data)

    descriptive_stats = data.describe(
        include="all"
    ).transpose()

    print("\nDESCRIPTIVE STATISTICS")
    print("-" * 45)
    print(descriptive_stats)

    descriptive_stats.to_csv(
        RESULT_DIR / "descriptive_statistics.csv"
    )

    # 1. Final grade distribution

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=data,
        x="G3",
        bins=11,
        kde=True
    )

    plt.title("Distribution of Final Student Grades")
    plt.xlabel("Final Grade (G3)")
    plt.ylabel("Number of Students")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "final_grade_distribution.png",
        dpi=300
    )
    plt.close()

    # 2. Study time versus final grade

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=data,
        x="studytime",
        y="G3"
    )

    plt.title("Study Time vs Final Grade")
    plt.xlabel("Study Time Category")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "study_time_vs_final_grade.png",
        dpi=300
    )
    plt.close()

    # 3. Average grade by study time

    average_grades = data.groupby(
        "studytime"
    )["G3"].mean()

    plt.figure(figsize=(8, 5))

    average_grades.plot(kind="bar")

    plt.title("Average Final Grade by Study Time")
    plt.xlabel("Study Time Category")
    plt.ylabel("Average Final Grade (G3)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "average_grade_by_study_time.png",
        dpi=300
    )
    plt.close()

    # 4. First grade versus final grade

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=data,
        x="G1",
        y="G3"
    )

    plt.title("First Period Grade vs Final Grade")
    plt.xlabel("First Period Grade (G1)")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "first_grade_vs_final_grade.png",
        dpi=300
    )
    plt.close()

    # 5. Correlation heatmap

    numeric_data = data.select_dtypes(
        include="number"
    )

    if numeric_data.shape[1] >= 2:

        plt.figure(figsize=(12, 9))

        sns.heatmap(
            numeric_data.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Heatmap of Numeric Features")
        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "correlation_heatmap.png",
            dpi=300
        )
        plt.close()

    print("\nEDA completed successfully.")
    print(f"Visualizations saved in: {CHART_DIR}")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    try:

        create_output_folders()

        print("Loading Student Performance dataset...")

        data = load_dataset()

        print("\nCleaning dataset...")

        data = clean_dataset(data)

        print(f"\nValid student records: {len(data)}")

        perform_eda(data)

        print("\nTraining Linear Regression model...")

        results = train_and_evaluate_model(
            data,
            RESULT_DIR,
            CHART_DIR
        )

        print("\n" + "=" * 55)
        print("MODEL EVALUATION RESULTS")
        print("=" * 55)

        for metric, value in results.items():

            if isinstance(value, (int, float, np.number)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        print("\nProject analysis completed successfully!")

    except Exception as error:

        print(f"\nERROR: {error}")
        raise


if __name__ == "__main__":
    main()
