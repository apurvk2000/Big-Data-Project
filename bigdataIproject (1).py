import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, plot_precision_recall_curve, plot_roc_curve
import matplotlib.pyplot as plt


def new_loan_state(record):
    if record.loan_status == 'Fully Paid' and record.recoveries == 0:
        record['loan_status'] = 0
    elif record.loan_status == 'Charged Off' and record.recoveries > 0:
        record['loan_status'] = 1
    else:
        record['loan_status'] = 2
    return record

# Read data
data=pd.read_csv('Lending_Club_Data.csv')
variables = pd.read_csv('Lending_Club_Dictionary.csv')['Variable']

# Data cleaning
data = data[variables]  # Very important!!!Drop useless and duplicated variables.
data = data.drop(['emp_title'], axis=1)  # This column is difficult to handle, so drop it
data = data.fillna(0)
for i in data.columns:  # For column in used to be str, convert 0 to '0'
    if data[i].dtypes == 'object':
        data[i] = data[i].astype('str')

#data = data.apply(lambda x:new_loan_state(x), axis=1)
data = data.drop(['recoveries'], axis=1)

# Numerating non-numerical data
encoder = preprocessing.OrdinalEncoder()
encoder.fit(data)
df = pd.DataFrame(encoder.transform(data),columns=data.columns)

# Random forest with default setting
time_start = time.time()
random_state = 809
X_train, X_test, y_train, y_test = train_test_split(df.drop(['loan_status'], axis=1),
                                                    df['loan_status'], test_size=0.3,
                                                    random_state=random_state)
rfc = RandomForestClassifier(random_state=random_state)
rfc.fit(X_train, y_train)
time_end = time.time()
print('time cost', time_end - time_start, 's')

confusion_matrix = confusion_matrix(y_test, rfc.predict(X_test))

