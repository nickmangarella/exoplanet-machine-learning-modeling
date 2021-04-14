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
After creating each model, the model with the worst score was the Linear Regression model with a score of 0.872. The model with the next highest score was the SVM model with a score of 0.874. The best model with the highest score was the Random Forest model with a score of 0.891. The Random Forest model proved to be the most accurate since it is designed to cross validate the nodes/subsets of random trees based on probability. Both the SVM model and Linear Regression model had very similar scores. SVM models create classification by plotting the data and clustering the points closest to one another. It sets a boundary about those clusters and looks at the distance of the other points to a boundary to determine their classification. Linear Regression models also plot the data but determine classification by the relationship between the mean of the targets and the set features. I believe the Random Forest model performed the best in this scenario because of cross validation, but based on the dataset many predictive models will perform well with this classification. What will set them apart is how well each model can be tuned.
