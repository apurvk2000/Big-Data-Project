import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# Read data
data = pd.read_csv('Lending_Club_Data.csv')

# Data cleaning
data = data.drop(
    ['emp_title', 'dti_joint', 'dti', 'dti_joint.1', 'int_rate', 'recoveries', 'total_pymnt', 'total_rec_int',
     'total_rec_late_fee', 'total_rec_prncp', 'tax_liens', 'verification_status_joint', 'verification_status',
     'num_bc_tl', 'num_tl_90g_dpd_24m', 'pub_rec_bankruptcies', 'revol_bal'], axis=1)
data = data.fillna(0)
for i in data.columns:  # For column in used to be str, convert 0 to '0'
    if data[i].dtypes == 'object':
        data[i] = data[i].astype('str')

# Numerating non-numerical data
encoder = preprocessing.OrdinalEncoder()
encoder.fit(data)
df = pd.DataFrame(encoder.transform(data), columns=data.columns)

# Data split and oversampling (SMOTE)
random_state = 809
X_train, X_test, y_train, y_test = train_test_split(df.drop(['loan_status'], axis=1),
                                                    df['loan_status'], test_size=0.3,
                                                    random_state=random_state)

oversample = SMOTE(random_state=random_state, k_neighbors=1, n_jobs=-1)
oversampled_X_train, oversampled_y_train = oversample.fit_resample(X_train, y_train)

# Parameter optimisation
n_estimators = [200, 500, 1000, 2000]
max_features = ['sqrt', 0.5]
max_depth = [10, 20, 50, 100]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
              'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
rfc = RandomForestClassifier(random_state=random_state, n_jobs=-1)
n_iter = 10  # Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
cv = 3  # Cross-validation folds
rf_random = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid, n_iter=n_iter, cv=cv,
                               random_state=random_state, n_jobs=-1, scoring='average_precision')
time_start = time.time()
rf_random.fit(oversampled_X_train, oversampled_y_train)
time_end = time.time()
print('time cost', time_end - time_start, 's')
best_parameters = rf_random.best_params_
print('The best Parameters are:')
for key, item in best_parameters.items():
    print(f'- {key}: {item}')

rfc_opt = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features = 'sqrt',
                                 max_depth=100, random_state=random_state, n_jobs=-1)
rfc_opt.fit(oversampled_X_train, oversampled_y_train)
print(rfc_opt.score(X_test, y_test))
confusion_matrix = confusion_matrix(y_test, rfc_opt.predict(X_test))
fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 10))
plot_roc_curve(rfc_opt, X_test, y_test, ax=ax)
plot_precision_recall_curve(rfc_opt, X_test, y_test, ax=ax1)
ax.set_title('ROC', fontsize=18)
ax1.set_title('Precision Recall', fontsize=18)
ax.legend(fontsize=15)
ax1.legend(fontsize=15)
plt.suptitle('LendingClub', fontsize=20)
plt.show()