# machine-learning-challenge

## Description
Using the Kepler Exoplanet data (https://www.kaggle.com/nasa/kepler-exoplanet-search-results) and Python: Pandas, Scikit-Learn, create multiple predictive models to classify candidate exoplants and analyze each models performance.

## Steps to Model Creation

### Step 1: Preprocess the raw data
* Cleaned the data, dropping null columns and rows
* Set features and target
* Scaled the data using `MinMaxScaler()` for Linear Regression and SVM models

### Step 2: Tune the models
* Performed hyperparameter tuning using `GridSearchCV()`
* Changed the parameter grid for each model
  * Linear Regression
  ```
  param_grid = {'C': [1, 5, 10],
              'penalty': ["l1", "l2"]}
  ```
  * SVM
  ```
  param_grid = {'C': [1, 5, 10],
              'gamma': [0.0001, 0.001, 0.01]}
  ```
  
  * Random Forest Classifier
  ```
  param_grid = {'n_estimators': [500, 1000],
              'max_features': ['auto', 'sqrt', 'log2']}
  ```

### Step 3: Evaluate model performance
* Obtained the best paremeters and best score from each `GridSearchCV()` of the trained models

## Analysis

