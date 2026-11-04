# Student Performance Analysis and Prediction System

## 1. Project Overview

The **Student Performance Analysis and Prediction System** is a Machine Learning project developed using Python.

The project analyses student academic data, identifies patterns that may affect academic performance, and predicts students' final Mathematics grades using a Linear Regression model.

The main objective is to understand student performance through data analysis, visualization, and machine learning techniques.

## 2. Dataset Description

This project uses the **Student Performance Dataset** from the UCI Machine Learning Repository.

- Dataset file: `student_performance_dataset.csv`
- Total records: 395
- Target variable: `G3` (Final Grade)
- Important input features: `G1`, `G2`, `studytime`, and `absences`
- Data format: CSV
- Separator: Semicolon (`;`)

The dataset contains students' academic and personal information that can be used to analyse and predict their final grades.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Visual Studio Code

## 4. Project Features

- Dataset loading and verification
- Data cleaning and preprocessing
- Missing-value analysis
- Duplicate record checking
- Descriptive statistical analysis
- Data visualization
- Correlation analysis
- Student performance analysis
- Linear Regression model training
- Hyperparameter tuning
- Model evaluation
- Actual vs. predicted grade comparison

## 5. Project Folder Structure

```text
Student-Performance-Analysis-and-Prediction-System/
│
├── data/
│   └── student_performance_dataset.csv
│
├── notebooks/
│   └── student_performance_analysis.ipynb
│
├── outputs/
│   ├── evaluation_results/
│   │   ├── actual_vs_predicted.csv
│   │   ├── dataset_overview.json
│   │   ├── descriptive_statistics.csv
│   │   ├── evaluation_metrics.csv
│   │   ├── hyperparameter_tuning_results.csv
│   │   └── missing_values_report.csv
│   │
│   └── visualizations/
│       ├── actual_vs_predicted.png
│       ├── average_grade_by_study_time.png
│       ├── bar_average_grade_by_studytime.png
│       ├── correlation_heatmap.png
│       ├── final_grade_distribution.png
│       ├── first_grade_vs_final_grade.png
│       ├── histogram_final_grades.png
│       └── study_time_vs_final_grade.png
│
├── screenshots/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   ├── dataset_overview.png
│   ├── evaluation_results.png
│   ├── grade_distribution.png
│   └── project_run.png
│
├── src/
│   ├── __init__.py
│   └── model.py
│
├── .gitignore
├── main.py
├── Project_Report.pdf
├── README.md
└── requirements.txt
```

## 6. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/avishekmoharana1-oss/Student-Performance-Analysis-and-Prediction-System.git
```

### Step 2: Open the Project Folder

```bash
cd Student-Performance-Analysis-and-Prediction-System
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the Project

```bash
python main.py
```

## 7. Machine Learning Methodology

The project follows these steps:

1. Load the student performance dataset.
2. Check and clean the dataset.
3. Analyse missing values and duplicate records.
4. Perform exploratory data analysis.
5. Generate visualizations to understand student performance.
6. Split the dataset into training and testing sets.
7. Build a Linear Regression model using Scikit-learn.
8. Perform hyperparameter tuning using GridSearchCV.
9. Evaluate the model using different performance metrics.
10. Save evaluation results and visualizations.

## 8. Model Evaluation

The Linear Regression model was evaluated using the following metrics:

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.6467 |
| Root Mean Squared Error (RMSE) | 2.3784 |
| R² Score | 0.7241 |
| Cross-Validation Mean R² | 0.8039 |
| Cross-Validation Standard Deviation | 0.0317 |

### Dataset Split

- Total records: 395
- Training records: 316
- Testing records: 79

These results represent the model's performance on the dataset used in this project.

## 9. Visualizations

The project generates visualizations to understand student academic performance, including:

- Final grade distribution
- Correlation heatmap
- Study time vs. final grade
- Average grade by study time
- First grade vs. final grade
- Actual vs. predicted grades

These visualizations help identify patterns and relationships in the dataset.

## 10. Limitations

- The model is trained using a limited dataset of 395 student records.
- Predictions depend on the quality and features of the available dataset.
- Actual student performance can be influenced by factors not included in the dataset.
- The model's results should not be treated as guaranteed academic outcomes.

## 11. Future Improvements

- Use larger and more diverse student datasets.
- Experiment with additional machine learning algorithms.
- Improve prediction accuracy through feature engineering.
- Develop a web-based application for student performance prediction.
- Add an interactive dashboard for better visualization.

## 12. Conclusion

The Student Performance Analysis and Prediction System demonstrates how Python and Machine Learning can be used to analyse student academic data and predict final grades.

The project includes data preprocessing, exploratory data analysis, visualization, Linear Regression model development, hyperparameter tuning, and model evaluation.

It provides practical experience in applying Machine Learning techniques to an educational dataset.

## 13. References

- UCI Machine Learning Repository – Student Performance Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
- Python Documentation: https://docs.python.org/3/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Scikit-learn Documentation: https://scikit-learn.org/stable/
