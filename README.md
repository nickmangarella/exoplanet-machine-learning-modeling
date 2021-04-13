# machine-learning-challenge

## Description
Utilize Python Pandas and Scikit-Learn to create multiple predictive models that classify candidate exoplanets from the Kepler Exoplanet data source (https://www.kaggle.com/nasa/kepler-exoplanet-search-results).

## Steps to Model Creation

### Step 1: Preprocess the raw data
* Cleaned the data, dropping null columns and rows
* Set features and targets
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
After creating each model, the model with the worst score was the Linear Regression model with a score of 0.66. The model with the next highest score was the SVM model with a score of 0.87. The best model with the highest score was the Random Forest decision tree model with a score of 0.89. Given the features and target, it is clear why the the best model was the Random Forest model. `RandomForestClassifier()`
