import numpy as np
from gabor import GaborData
from data import DataHeadposeDataset
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.metrics import mean_squared_error
import joblib

def gridSearch():
  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'bootstrap': bootstrap}
  pprint(random_grid)
  
  data = Data()
  X_train,y_train, X_test,y_test = data(path)
  X_train = np.vstack(X_train)
  y_train = np.vstack(y_train)
  X_test = np.vstack(X_test)  
  # Use the random grid to search for best hyperparameters
  # First create the base model to tune
  rf = RandomForestRegressor()
  # Random search of parameters, using 3 fold cross validation, 
  # search across 100 different combinations, and use all available cores
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
  # Fit the random search model
  rf_random.fit(X_train, y_train)
  print(rf_random.best_params_)


def main(path=None):
  data = Data()
  X_train,y_train, X_test,y_test = data(path)
  X_train = np.vstack(X_train)
  y_train = np.vstack(y_train)
  X_test = np.vstack(X_test)
  regr = RandomForestRegressor(n_estimators=1000, random_state=0,min_samples_split=2,min_samples_leaf=1,\
                            max_features= 'sqrt',max_depth= 20,bootstrap= True)
  regr.fit(X_train, y_train.flatten())
  y_ = regr.predict(X_test)
  mse = mean_squared_error(y_test, y_)
  print("::MSE::{}".format(mse))
  print("Saving the model")
  joblib.dump(regr, "./random_forest.joblib")
