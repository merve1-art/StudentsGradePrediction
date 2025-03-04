# Predicting Student Academic Performance Using Random Forest Classification

**Description:**

This project explores the factors influencing student academic performance using the "Student Alcohol Consumption" dataset from Kaggle. It employs Random Forest Classification to predict student grade categories and identifies the most significant features contributing to academic success. The analysis includes data preprocessing, model training, hyperparameter tuning, feature selection, and SHAP (SHapley Additive exPlanations) analysis for feature impact interpretation.

**Dataset:**

* **Source:** [Student Alcohol Consumption Dataset](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption) from Kaggle.
* **Content:** The dataset contains information about student demographics, social habits, and academic performance, including math and Portuguese language grades.
* **Preprocessing:** The dataset was cleaned, merged, and preprocessed, including outlier removal, feature engineering, and one-hot encoding for categorical variables.

**Methodology:**

1.  **Data Acquisition and Merging:**
    * Downloaded and combined math and Portuguese language student datasets.
2.  **Exploratory Data Analysis (EDA):**
    * Checked data types, missing values, and summary statistics.
    * Created a new "average_grade" feature and calculated the average grade of each student in one semester.
    * Visualized data distributions and identified outliers.
3.  **Data Preprocessing:**
    * Removed outliers using the IQR method.
    * Removed grades for each exam and stuck with the average grade only.
    * One-hot encoded categorical features.
    * Converted grade categories into numerical labels.
    * Handled class imbalance using SMOTE.
4.  **Model Training and Evaluation:**
    * Trained and evaluated multiple classification models (Logistic Regression, Random Forest, SVC, KNN, Decision Tree).
    * Observed low initial accuracy due to grade category distribution.
5.  **K-Means Clustering:**
    * Used K-Means clustering to identify optimal grade categories (K=3).
    * Mapped clusters to "Low," "Medium," and "High" grade categories.
    * Improved model accuracy with the new grade categories.
6.  **Feature Selection:**
    * Performed recursive feature elimination using Random Forest to identify the most important features.
    * Selected top 16 and 36 features for further analysis.
7.  **Hyperparameter Tuning:**
    * Used RandomizedSearchCV to tune hyperparameters for the Random Forest model.
    * Evaluated model performance with the selected features.
8.  **Feature Impact Analysis:**
    * Visualized feature importances using bar plots.
    * Used SHAP values to explain how individual features impact grade predictions.

**Results:**

* The Random Forest Classifier achieved an accuracy of approximately 72% with both 16 and 36 features.
* Feature selection did not significantly impact accuracy, suggesting that the reduced feature set is sufficient.
* Features like "absences," "failures," and "health" were identified as the most influential factors.
* SHAP analysis provided insights into how individual features affect the probability of students falling into different grade categories.

**How to Run the Code:**

1.  Clone the repository: `git clone [repository URL]`
2.  Install required dependencies: `pip install pandas numpy scikit-learn imblearn shap matplotlib seaborn kaggle`
3.  Download the dataset from Kaggle using the Kaggle API and place it in the project directory.
4.  Run the Python script: `python your_script_name.py`

**Files Included:**

* `students_grade_classification.py`: Python script containing the project code.
* `student-mat.csv`: Math student dataset.
* `student-por.csv`: Portuguese student dataset.
* `README.md`: Project documentation.

**Dependencies:**

* pandas
* numpy
* scikit-learn
* imblearn
* shap
* matplotlib
* seaborn
* kaggle

**Author:**

* Merve Tuncer Ozer
* [My GitHub Profile URL] (https://github.com/merve1-art)

**License:**

* [MIT License]

**Future Work:**

* Further investigate and predict the grades with regression models instead of classification.
* Investigate the impact of other features not included in the dataset.
* Develop a web application to visualize the results and provide interactive insights.
