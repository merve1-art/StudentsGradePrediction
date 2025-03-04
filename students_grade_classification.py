# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:39:58 2025

@author: meerv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Scikit-learn utilities
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

# Machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

# Handling imbalanced data
from imblearn.over_sampling import SMOTE



#PART 1:
# Load the datasets

# This script provides two options for loading the dataset. 
# Please refer to the README file for step-by-step instructions on how to download the dataset.

##########################################################

# Option 1: Kaggle API Download
# Kaggle dataset URL
dataset_url = "https://www.kaggle.com/datasets/uciml/student-alcohol-consumption"

# Download dataset using opendatasets library
od.download(dataset_url)

# Load datasets from the downloaded folder
df_mat = pd.read_csv("student-alcohol-consumption/student-mat.csv")  
df_por = pd.read_csv("student-alcohol-consumption/student-por.csv")  

# Check if both DataFrames have the same columns before merging
if list(df_mat.columns) == list(df_por.columns):  
    df_combined = pd.concat([df_mat, df_por], axis=0, ignore_index=True)
    print(f"Combined dataset shape: {df_combined.shape}")  # Expected output: (980, number of columns)
else:
    print("Error: The datasets have different columns and cannot be combined directly.")
####################################################################



####################################################################
# Option 2: Manual Download
# If you manually downloaded the dataset, ensure that 'student-mat.csv' and 'student-por.csv' are in the project root directory.
df_mat = pd.read_csv("student-mat.csv")  
df_por = pd.read_csv("student-por.csv")  

# Merge datasets if both have the same structure
if list(df_mat.columns) == list(df_por.columns):  
    df_combined = pd.concat([df_mat, df_por], axis=0, ignore_index=True)
    print(f"Combined dataset shape: {df_combined.shape}")  # Expected output: (980, number of columns)
else:
    print("Error: The datasets have different columns and cannot be combined directly.")

#######################################################################3
#PART 2:

# Exploratory Data Analysis (EDA)

# 1) Check the data types and missing values
print("\n--- Data Types and Missing Values ---")
df_combined.info()

# 2) Summary statistics for numeric variables
print("\n--- Summary Statistics for Numeric Variables ---")
print(df_combined.describe())

# 3) Distribution of categorical variables
print("\n--- Distribution of Categorical Variables ---")
categorical_columns = df_combined.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\n{col} distribution:\n{df_combined[col].value_counts()}")

# 4) Creating a new variable: 'average_grade'
df_combined['average_grade'] = df_combined[['G1', 'G2', 'G3']].mean(axis=1)
df_combined.drop(columns=['G1', 'G2', 'G3'], inplace=True)

# 5) Visualization for Numeric Variables

# Define numeric columns for visualization
numeric_columns = ['age', 'absences', 'average_grade']

# Histograms
plt.figure(figsize=(12, 6))
df_combined[numeric_columns].hist(figsize=(12, 6), bins=20, edgecolor='black')
plt.suptitle('Distribution of Numeric Variables', fontsize=14)
plt.show()

# Boxplots for Outlier Detection
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df_combined[col], color='skyblue')
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# 6) Removing Outliers Using IQR (for Skewed Data)
def remove_outliers_iqr(df, column):
    """Removes outliers based on the Interquartile Range (IQR) method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal for 'absences' and 'average_grade' (both are skewed)
df_cleaned = remove_outliers_iqr(df_combined, "absences")
#df_cleaned = remove_outliers_iqr(df_cleaned, "average_grade")

# Boxplots After Outlier Removal
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df_cleaned[col], color='lightcoral')
    plt.title(f"Boxplot of {col} After Outlier Removal")
plt.tight_layout()
plt.show()

#PART 3:  Data Preprocessing 


# Step 1: Clone the cleaned data for further processing
df_combined_score=df_cleaned.copy()

# Step 2: Overview of the target variable (average_grade) statistics
print(df_combined_score['average_grade'].describe())
"""
count    990.000000
mean      11.316835
std        3.235618
min        1.333333
25%        9.333333
50%       11.333333
75%       13.333333
max       19.333333
"""
# Step 3: Convert 'average_grade' into categories based on percentiles
# Define bins and labels for categorization
bins = [0, 9.33, 11.33, 13.33, 20]
labels = ["Low", "Medium", "High", "Best"]

# Apply binning to categorize grades
df_combined_score["grade_category"] = pd.cut(df_combined_score["average_grade"], bins=bins, labels=labels, right=False)

# Check for missing categories and handle them
df_combined_score.loc[df_combined_score["grade_category"].isna(), "average_grade"]

# Convert grade categories to numeric encoding for machine learning models
df_combined_score["grade_category_encoded"] = df_combined_score["grade_category"].astype("category").cat.codes

# Step 4: One-hot encode categorical features
binary_columns = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 
    'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'
]

# Apply one-hot encoding to binary categorical columns
df_combined_score = pd.get_dummies(df_combined_score, columns=binary_columns, drop_first=True)

# One-hot encode non-binary categorical variables
df_combined_score = pd.get_dummies(df_combined_score, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)

# Step 5: Convert non-categorical columns to integer type
for col in df_combined_score.select_dtypes(include=["bool", "int", "float"]).columns:
    df_combined_score[col] = df_combined_score[col].astype(int)
    
    
    
    
# PART 4: Machine Learning Classification Function

def machine_learning_classification(models, data, target_variable='grade_category_encoded', test_size=0.2, random_state=42):
    """
    Train and evaluate multiple classification models.

    Parameters:
    models (list): List of initialized classification models.
    data (DataFrame): The dataset containing features and target.
    target_variable (str): The column name of the target variable.
    test_size (float): Proportion of the data to be used for testing.
    random_state (int): Random state for reproducibility.

    Returns:
    dict: A dictionary containing evaluation metrics for each model.
    """
    
    # Split data into features (X) and target (y)
    X = data.drop(columns=["average_grade", "grade_category", "grade_category_encoded"])  # Features
    y = data[target_variable]  # Target

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, random_state=random_state, stratify=y_resampled)

    # Store results for each model
    results = {}

    # Initialize scaler (for models requiring scaling)
    scaler = StandardScaler()
    models_needing_scaling = ['LogisticRegression', 'SVC', 'KNeighborsClassifier']

    # Loop through models to train and evaluate
    for model in models:
        model_name = model.__class__.__name__

        # Check if the model requires scaling
        if model_name in models_needing_scaling:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
            X_test_scaled = X_test  # No scaling applied

        # Make predictions and compute evaluation metrics
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Store results
        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': conf_matrix
        }

    return results

# PART 5: Model Definition and Evaluation

# List of models to evaluate
models = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier()
]

# Evaluate the models
results = machine_learning_classification(models, df_combined_score, target_variable='grade_category_encoded')

# Print results for each model
for model, metrics in results.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print("\n")

"""
  LogisticRegression:
    Accuracy: 0.49099099099099097
    Precision: 0.4937621444974386
    Recall: 0.49099099099099097
    F1 Score: 0.48483948779079133
    Confusion Matrix:
   [[34  8  9  5]
   [ 9 17 11 18]
   [ 5  8 22 21]
   [ 3  7  9 36]]


  RandomForestClassifier:
    Accuracy: 0.4774774774774775
    Precision: 0.4594194292163463
    Recall: 0.4774774774774775
    F1 Score: 0.46569331046138895
    Confusion Matrix: 
   [[39  6  8  3]
   [16 12 12 15]
   [ 8 16 22 10]
   [ 4 10  8 33]]


  SVC:
    Accuracy: 0.5315315315315315
    Precision: 0.5253669448057888
    Recall: 0.5315315315315315
    F1 Score: 0.5257369544734254
    Confusion Matrix: 
   [[38  6  9  3]
   [11 16 13 15]
   [ 3 13 29 11]
   [ 2  7 11 35]]


  KNeighborsClassifier:
    Accuracy: 0.4774774774774775
    Precision: 0.47504848817962425
    Recall: 0.4774774774774775
    F1 Score: 0.47441158224846763
    Confusion Matrix:
   [[34  7 12  3]
   [15 17 13 10]
   [ 6 14 27  9]
   [ 7  8 12 28]]


  DecisionTreeClassifier:
    Accuracy: 0.509009009009009
    Precision: 0.5021211205053296
    Recall: 0.509009009009009
    F1 Score: 0.5026996320814356
    Confusion Matrix:
   [[37 11  6  2]
   [13 20 11 11]
   [ 9 11 23 13]
   [ 9  5  8 33]]

"""

# Additional observation: The accuracy is low, possibly due to the choice of 4
# categories. Reconsidering the number of categories may improve performance.


# PART 6: K-Means Clustering to Find the Best Category Number

# Step 1: Use the elbow method to find the optimal number of clusters (K)

inertia = []
K_range = range(2, 6)  # Try 2 to 5 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_combined_score[["average_grade"]])  # Use grade-related column
    inertia.append(kmeans.inertia_)

# Plot elbow method to determine best cluster count
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clustering')
plt.show()




# Step 2: Evaluate different K-values for KNN (K-Nearest Neighbors) to identify the best k
X = df_combined_score.drop(columns=["average_grade", "grade_category", "grade_category_encoded"])  # Features
y = df_combined_score["grade_category_encoded"]  # Target variable

# Loop through different k-values and check performance using cross-validation
for k in range(2, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    print(f"K={k}, Accuracy={scores.mean():.4f}")
"""
K=2, Accuracy=0.3061
K=3, Accuracy=0.3424
K=4, Accuracy=0.3475
K=5, Accuracy=0.3495
Based on the elbow method and KNN evaluation, we select k=3 for clustering.
"""
# Step 3: Apply K-Means with K=3 for clustering the grades into 3 categories

grades = df_combined_score[['average_grade']].values  # Reshape data for K-Means

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_combined_score['grade_category_kmeans'] = kmeans.fit_predict(grades)


#Step 4: Visualization of clusters 

# Sort dataframe for a better visual representation
df_sorted = df_combined_score.sort_values(by="average_grade")

# Plot the grades colored by their assigned category
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_sorted, x="average_grade", y=np.zeros(len(df_sorted)), 
                hue=df_sorted['grade_category_kmeans'], palette="viridis", alpha=0.7)

# Plot K-Means cluster centers
centroids = kmeans.cluster_centers_
plt.scatter(centroids, [0] * len(centroids), color="red", marker="X", s=200, label="Centroids")

# Labels and title
plt.xlabel("Average Grade")
plt.ylabel("")
plt.yticks([])  # Remove y-axis ticks as they are not needed
plt.title("K-Means Clustering of Grades")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()


# Step 5: Map clusters to grade categories

# Compute the mean of each cluster and sort by value
cluster_means = df_combined_score.groupby('grade_category_kmeans')['average_grade'].mean()
sorted_clusters = cluster_means.sort_values().index  # Sorted clusters by mean grade

# Create a mapping: lowest mean → 0 (Low), middle mean → 1 (Medium), highest mean → 2 (High)
cluster_mapping = {sorted_clusters[0]: 'Low', sorted_clusters[1]: 'Medium', sorted_clusters[2]: 'High'}
df_combined_score['grade_category'] = df_combined_score['grade_category_kmeans'].map(cluster_mapping)



# Step 5: Visualize the distribution of grade categories
grade_counts = df_combined_score['grade_category'].value_counts()

# Plot the counts of each category
plt.figure(figsize=(6, 4))
sns.barplot(x=grade_counts.index, y=grade_counts.values, palette='viridis')
plt.xlabel('Grade Category')
plt.ylabel('Count')
plt.title('Distribution of Grade Categories')
plt.show()


# Step 6: Encode the categorical labels as numbers
grade_encoding = {"Low": 0, "Medium": 1, "High": 2}
df_combined_score['grade_category_encoded'] = df_combined_score['grade_category'].map(grade_encoding)

# Verify the changes
print(df_combined_score[['average_grade', 'grade_category_kmeans', 'grade_category', 'grade_category_encoded']].head())

# Drop the temporary 'grade_category_kmeans' column
df_combined_score = df_combined_score.drop(columns="grade_category_kmeans")


# Step 7: Apply classification models to evaluate performance
results = machine_learning_classification(models, df_combined_score, target_variable='grade_category_encoded')

# Print the results of each model
for model, metrics in results.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print("\n")
"""
LogisticRegression:
  Accuracy: 0.5728813559322034
  Precision: 0.5721213939303035
  Recall: 0.5728813559322034
  F1 Score: 0.5706993677892302
  Confusion Matrix: [[64 16 18]
 [20 44 34]
 [14 24 61]]


RandomForestClassifier:
  Accuracy: 0.7220338983050848
  Precision: 0.7154954954954954
  Recall: 0.7220338983050848
  F1 Score: 0.7132588880737336
  Confusion Matrix: [[83 11  4]
 [24 49 25]
 [ 4 14 81]]


SVC:
  Accuracy: 0.688135593220339
  Precision: 0.6886998734518599
  Recall: 0.688135593220339
  F1 Score: 0.6863602529952857
  Confusion Matrix: [[75 13 10]
 [14 54 30]
 [ 6 19 74]]


KNeighborsClassifier:
  Accuracy: 0.6677966101694915
  Precision: 0.6663802277166423
  Recall: 0.6677966101694915
  F1 Score: 0.6530319842105592
  Confusion Matrix: [[87  3  8]
 [30 41 27]
 [12 18 69]]


DecisionTreeClassifier:
  Accuracy: 0.6949152542372882
  Precision: 0.6943347059635687
  Recall: 0.6949152542372882
  F1 Score: 0.6929288567475707
  Confusion Matrix: [[80 12  6]
 [20 60 18]
 [11 23 65]]
"""

#The accuracy is improved especially for the random forest reressor.
# The next step might be removing the least significant variables
#to make model more simple but still powerfull.






# PART 7: Remove the least significant variables with random forest regression:


# Select features and target
X = df_combined_score.drop(columns=["average_grade","grade_category", "grade_category_encoded"])  # Features
y = df_combined_score["grade_category_encoded"]  # Target


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)  # Resampling

# Split into training and testing sets (use stratify=y_resampled)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

        
# Store results
feature_counts = []
accuracy_scores = []

# Track current feature set
best_features = list(X.columns)

# Feature selection loop
while len(best_features) > 1:
    # Compute feature importances
    feature_importances = pd.Series(clf.feature_importances_, index=best_features).sort_values(ascending=False)

    # Remove the least important feature
    least_important = feature_importances.idxmin()
    best_features.remove(least_important)  # Remove it from feature list

    # Subset dataset with selected features
    X_selected = X_resampled[best_features]

    # Split into training and testing sets (use stratify=y_resampled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Train Random Forest with reduced features
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate new model
    y_pred = clf.predict(X_test)
    new_accuracy = accuracy_score(y_test, y_pred)

    # Store feature count and accuracy for plotting
    feature_counts.append(len(best_features))
    accuracy_scores.append(new_accuracy)

    print(f"Features left: {len(best_features)} | Accuracy: {new_accuracy:.4f}")

# 🔹 Plot Number of Features vs. Accuracy
plt.figure(figsize=(10, 6))
plt.plot(feature_counts, accuracy_scores, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Feature Selection: Number of Features vs. Accuracy")
plt.gca().invert_xaxis()  # Makes sure decreasing features is left-to-right
plt.grid(True)
plt.show()
#Based on the features vs accuracy, the model gives the highest accuracy with
#36 features with around 0.75 accuracy. However although the accuracy drop
#sligtly, the model gives 0.74 accuracy with 16 features too. Therefore I will
#I will try the model with 36 and 16 features.


# PART 8: Modelling the data with 16 and 36 most significant variables with random forest regression:
    
# Define the number of top features to evaluate
number_of_features = [36, 16]

# Select features and target
X = df_combined_score.drop(columns=["average_grade", "grade_category", "grade_category_encoded"])  # Features
y = df_combined_score["grade_category_encoded"]  # Target

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train an initial Random Forest model to get feature importances
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_resampled, y_resampled)

# Get feature importance ranking
feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Iterate over different feature subsets
for i in number_of_features:
    print(f"\n=== Training Model with Top {i} Features ===")

    # Select the top i features
    top_features = feature_importances.nlargest(i).index.tolist()
    
    # Filter dataset to only include selected features
    X_selected = X[top_features]

    # Apply SMOTE again
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # Define hyperparameter search space
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Perform Randomized Search for hyperparameter tuning
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_dist, 
        n_iter=10,  # Reduce iterations to speed up training
        cv=5, 
        n_jobs=-1, 
        random_state=42, 
        scoring='accuracy'
    )
    random_search.fit(X_train, y_train)

    # Store best hyperparameters for this feature set
    best_params = random_search.best_params_
    print(f"Best hyperparameters for {i} features: {best_params}")

    # Train a new model using the best hyperparameters
    best_clf = RandomForestClassifier(**best_params, random_state=42)
    best_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = best_clf.predict(X_test)

    # Evaluate accuracy
    best_accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print(f"Initial Accuracy for the top {i} features: {best_accuracy:.4f}")  
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print("=" * 50)
"""
=== Training Model with Top 36 Features ===
Best hyperparameters for 36 features: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 20, 'bootstrap': True}
Initial Accuracy for the top 36 features: 0.7186

Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.85      0.80        98
           1       0.66      0.51      0.57        98
           2       0.72      0.80      0.76        99

    accuracy                           0.72       295
   macro avg       0.71      0.72      0.71       295
weighted avg       0.71      0.72      0.71       295


Confusion Matrix:
[[83 10  5]
 [22 50 26]
 [ 4 16 79]]
==================================================

=== Training Model with Top 16 Features ===
Best hyperparameters for 16 features: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None, 'bootstrap': False}
Initial Accuracy for the top 16 features: 0.7186

Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.83      0.81        98
           1       0.68      0.52      0.59        98
           2       0.68      0.81      0.74        99

    accuracy                           0.72       295
   macro avg       0.72      0.72      0.71       295
weighted avg       0.72      0.72      0.71       295


Confusion Matrix:
[[81  9  8]
 [17 51 30]
 [ 4 15 80]]
==================================================
36 Features: 71.86%
16 Features: 71.86%
🔹 Observation: Reducing features from 36 to 16 did not decrease the accuracy.
 This suggests that the removed features did not add significant predictive power 
 and the model is just as good with fewer features.
🔹 Implication: A smaller feature set can improve model interpretability, training
 speed, and avoid overfitting while maintaining performance.

"""


#STEP 9: Which feature influence the grades how: 
    
# Get feature importances
importances = best_clf.feature_importances_
feature_names = X_resampled.columns  # X should only contain the selected features
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.show()


#Based on this figure, features like absences, failures, health,...,
#influence the grade class at most. However, it does not say exactly,
#how these features influence the grade prediction.

#To find out how features influence the grades, we will use SHAP.

# Create an explainer using the trained model
explainer = shap.TreeExplainer(best_clf)

# Compute SHAP values for test data
shap_values = explainer.shap_values(X_test)

# Summary plot to show overall feature importance
shap.summary_plot(shap_values, X_test)


# List of classes to iterate over
class_index = [0, 1, 2]

# Loop over features
for i in feature_names:
    # Create a figure with 3 subplots (one for each class)
    plt.figure(figsize=(15, 5))  
    
    # Loop over classes and plot them on separate subplots
    for j, class_id in enumerate(class_index):
        shap_values_selected = shap_values[:, :, class_id]  # Shape becomes (295, 16)
        
        # Generate SHAP dependence plot for this class using a new axis (sub-plot)
        ax = plt.subplot(1, 3, j + 1)  # 1 row, 3 columns, the (j+1)-th subplot
        
        shap.dependence_plot(i, shap_values_selected, X_test, show=False, ax=ax)  # Pass `ax=ax` to plot on this axis
        
        # Add title for each subplot to specify the class
        ax.set_title(f"Class {class_id}")
        
    # Add a general title for the entire figure
    plt.suptitle(f"SHAP Dependence Plot for Feature {i}", fontsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make space for the suptitle
    
    # Show the plot for this feature
    plt.show()





