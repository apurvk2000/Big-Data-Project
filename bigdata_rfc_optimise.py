import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import preprocessing


# Read data
data=pd.read_csv('Lending_Club_Data.csv')
variables = pd.read_csv('Lending_Club_Dictionary.csv')['Variable']

# Data cleaning
data = data[variables]  # Very important!!!Drop useless and duplicated variables.
data = data.drop(['emp_title',
                  'dti',
                  'dti_joint',
                  'recoveries',
                  'int_rate'
                  'total_pymnt',
                  'total_rec_int',
                  'total_rec_late_fee',
                  'total_rec_prncp'], axis=1)  # This column is difficult to handle, so drop it
data = data.fillna(0)
for i in data.columns:  # For column in used to be str, convert 0 to '0'
    if data[i].dtypes == 'object':
        data[i] = data[i].astype('str')

# Numerating non-numerical data
encoder = preprocessing.OrdinalEncoder()
encoder.fit(data)
df = pd.DataFrame(encoder.transform(data), columns=data.columns)

random_state = 809
X_train, X_test, y_train, y_test = train_test_split(df.drop(['loan_status'], axis=1),
                                                    df['loan_status'], test_size=0.3,
                                                    random_state=random_state)

# Grid search parameter optimisation
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = [0.5, 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}

rfc = RandomForestClassifier()
n_iter = 20  # Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
cv = 3  # Cross-validation folds
n_jobs = -1  # Uses all cores available
verbose = 5  # Controls how much the algo will print text to inform us of what is going on
rf_random = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid, n_iter=n_iter, cv=cv, verbose=verbose, random_state=random_state, n_jobs=n_jobs)
time_start = time.time()
rf_random.fit(X_train, y_train)
time_end = time.time()
print('time cost', time_end - time_start, 's')
best_parameters = rf_random.best_params_
print('The best Parameters are:')
for key, item in best_parameters.items():
    print(f'- {key}: {item}')