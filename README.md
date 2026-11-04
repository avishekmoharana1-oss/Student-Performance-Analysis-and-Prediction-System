# Student Performance Analysis and Prediction System

## 1. Project Overview

The Student Performance Analysis and Prediction System is a machine learning project developed using Python. It analyses student academic data, identifies performance patterns, and predicts final Mathematics grades using a Linear Regression model.

The project includes data preprocessing, exploratory data analysis, data visualization, model training, prediction, hyperparameter tuning, and performance evaluation.

## 2. Dataset Description

This project uses the Student Performance Mathematics dataset from the UCI Machine Learning Repository.

- **Dataset file:** `data/student_performance_dataset.csv`
- **Dataset source:** UCI Student Performance Dataset
- **Target variable:** `G3` (Final Mathematics Grade)
- **Dataset size:** 395 student records and 33 columns
- **Grade range:** 0 to 20

The dataset contains demographic characteristics, family background, study habits, school-related information, and academic performance.

The model uses available features other than `G3` to predict the final grade. These include earlier examination grades `G1` and `G2`.

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

- Dataset loading and inspection
- Data cleaning and duplicate checking
- Missing-value analysis
- Descriptive statistical analysis
- Exploratory Data Analysis (EDA)
- Data visualization
- Histogram and bar chart generation
- Scatter plot and box plot generation
- Correlation heatmap
- Linear Regression model training
- Data preprocessing using Scikit-learn pipelines
- Hyperparameter tuning using GridSearchCV
- Five-fold cross-validation
- Model evaluation using MAE, RMSE, and R²
- Actual-versus-predicted grade comparison
- Exporting evaluation results and visualizations

## 5. Project Structure

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

## 6. Installation and Execution

### Step 1: Install Python

Install Python if it is not already installed.

### Step 2: Clone the Repository

```bash
git clone https://github.com/avishekmoharana1-oss/Student-Performance-Analysis-and-Prediction-System.git
```

### Step 3: Open the Project

Open the downloaded project folder in Visual Studio Code.

### Step 4: Create a Virtual Environment

Open the VS Code terminal and run:

```bash
python -m venv venv
```

### Step 5: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 7: Run the Main Python Program

```bash
python main.py
```

### Step 8: Open the Jupyter Notebook

Open the following file in Visual Studio Code:

`notebooks/student_performance_analysis.ipynb`

Select the project's Python environment as the notebook kernel and execute the cells in order.

If Jupyter is not installed, run:

```bash
pip install jupyter ipykernel
```

### Step 9: View the Results

After successful execution, check:

- `outputs/visualizations/` — generated charts.
- `outputs/evaluation_results/` — evaluation metrics, descriptive statistics, missing-value report, tuning results, dataset overview, and prediction comparisons.

## 7. Machine Learning Methodology

The project uses Linear Regression to predict students' final Mathematics grades.

- **Input:** Available student-related and academic features.
- **Target:** `G3` (Final Mathematics Grade).
- **Algorithm:** Linear Regression.

The dataset is divided into training and testing sets using an 80:20 split.

A Scikit-learn preprocessing pipeline is implemented:

- Numerical features are handled using median imputation.
- Categorical features are handled using most-frequent imputation and one-hot encoding.
- Preprocessing is fitted within the training workflow to reduce the risk of data leakage.

Hyperparameter tuning is performed using GridSearchCV to select the best `fit_intercept` setting for Linear Regression.

Five-fold cross-validation is used on the training data.

## 8. Model Evaluation

The model is evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted grades.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalising larger errors more strongly.
- **R-squared (R²):** Measures the proportion of variation in final grades explained by the model.
- **Cross-validation:** Evaluates model performance across multiple training-data splits.

The project also saves actual and predicted grades for comparison.

## 9. Results

The following results were obtained from the successful project execution.

| Metric | Result |
|---|---:|
| Student records analysed | 395 |
| Training records | 316 |
| Testing records | 79 |
| MAE | 1.6467 |
| RMSE | 2.3784 |
| Test R² | 0.7241 |
| Cross-validation mean R² | 0.8039 |
| Cross-validation standard deviation | 0.0317 |
| Best tuning CV RMSE | 2.0195 |
| Best `fit_intercept` setting | True |

## 10. Key Findings

- The correlation between second-period grades (`G2`) and final grades (`G3`) is 0.905, indicating a strong positive relationship.
- The correlation between absences and final grades (`G3`) is 0.034, indicating almost no linear relationship.
- Study-time categories 3 and 4 show higher average final grades than categories 1 and 2.
- The average final grades for study-time categories 1, 2, 3, and 4 are 10.05, 10.17, 11.40, and 11.26 respectively.
- Earlier examination grades are useful predictors of final grades in this dataset.

These findings describe associations and do not establish cause and effect.

## 11. Limitations

- The dataset contains only 395 student records.
- The dataset represents students from two Portuguese schools, so results may not generalise to all schools or regions.
- Study time is recorded in categories rather than exact hours.
- The dataset provides the number of school absences but does not contain an attendance-percentage field.
- Earlier grades `G1` and `G2` are used as predictors. Therefore, the model is more suitable when these grades are already available.
- Linear Regression may not capture complex non-linear relationships.
- The model has not been tested on a separate external dataset.
- Predictions are estimates and are not guaranteed examination results.

## 12. Future Improvements

- Compare Linear Regression with other suitable machine learning algorithms.
- Use larger and more diverse student datasets.
- Explore additional feature engineering techniques.
- Perform more extensive hyperparameter tuning.
- Evaluate the model using additional external datasets.
- Develop a user-friendly web interface for entering student information and viewing predictions.
- Investigate approaches to keep predicted grades within the valid 0–20 range.

## 13. Conclusion

This project demonstrates how Python, data analysis, visualization, and machine learning can be used to analyse student academic performance and predict final Mathematics grades.

The project uses the UCI Student Performance Mathematics dataset containing 395 records and 33 columns. Data preprocessing and exploratory data analysis were performed to understand the dataset and identify academic performance patterns.

A Linear Regression model was developed to estimate final Mathematics grades. The model achieved an MAE of 1.6467, an RMSE of 2.3784, and an R² score of 0.7241 on the test dataset.

Overall, the project provided practical experience in data cleaning, exploratory data analysis, data visualization, machine learning model development, prediction, and evaluation.

## 14. References

1. UCI Machine Learning Repository — Student Performance Dataset
2. Scikit-learn Documentation
3. Pandas Documentation
4. NumPy Documentation
5. Matplotlib Documentation
6. Seaborn Documentation
