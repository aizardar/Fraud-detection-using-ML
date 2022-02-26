import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, RocCurveDisplay, ConfusionMatrixDisplay

from sklearn import tree


# Read data
print("\nReading data\n")
features = pd.read_csv("data-new/transactions_obf.csv")
labels = pd.read_csv("data-new/labels_obf.csv")


# Create new column where Fraud is denoted by 1. 
labels['Fraud'] = 1


# Merge features and labels on eventId 
df = pd.merge(features, labels, on="eventId", how = "left", validate = 'many_to_one')


# Label normal transactions with 0
df.Fraud.fillna(0, inplace = True)
df['Fraud'] = df['Fraud'].astype(int)


#Drop columns that are not useful
df.drop(['eventId', 'reportedTime'], axis = 1, inplace = True)


# Look if there are any duplicate entries
df.duplicated().sum()


# Feature engineering

# Time

print("\nPerforming feature engineering\n")

# Convert transactionTime column into datetime format
df['transactionTime']= pd.to_datetime(df['transactionTime'])

# Extract year, month, day, day of the week, and hour from the transactionTime column
df['year'] = pd.DatetimeIndex(df['transactionTime']).year
df['month'] = pd.DatetimeIndex(df['transactionTime']).month
df['day'] = pd.DatetimeIndex(df['transactionTime']).day
df['day_of_week'] = pd.DatetimeIndex(df['transactionTime']).dayofweek
df['hour'] = pd.DatetimeIndex(df['transactionTime']).hour


# Sort data according to account number and transaction time. 
df = df.sort_values(['accountNumber','transactionTime'])


# Extract new feature based on the minutes since last transaction was made from a particular customer account

df['days_since_last_trans'] = (df.groupby('accountNumber')['transactionTime'].diff().dt.days)
df['hours_since_last_trans'] = df['days_since_last_trans']*24 + (df.groupby('accountNumber')
                                ['transactionTime'].diff().dt.components['hours'])
df['min_since_last_trans'] = df['hours_since_last_trans']*60 + (df.groupby('accountNumber')
                                  ['transactionTime'].diff().dt.components['minutes'])

#  Drop days and hours as this information is already contained in minutes!

df = df.drop(['days_since_last_trans','hours_since_last_trans'], axis = 1)


# Extract new feature based on the minutes since last transaction was made countrywise from a particular customer account

df['days_since_last_trans_country_pos'] = (df.groupby(['accountNumber','merchantCountry', 'posEntryMode'])['transactionTime'].diff().dt.days)
df['hours_since_last_trans_country_pos'] = df['days_since_last_trans_country_pos']*24 + (df.groupby(['accountNumber','merchantCountry','posEntryMode'])
                                 ['transactionTime'].diff().dt.components['hours'])
df['min_since_last_trans_country_pos'] = df['hours_since_last_trans_country_pos']*60 + (df.groupby(['accountNumber','merchantCountry','posEntryMode'])
                                   ['transactionTime'].diff().dt.components['minutes'])

#  Drop days and hours as this information is already contained in minutes!

df = df.drop(['days_since_last_trans_country_pos','hours_since_last_trans_country_pos'], axis = 1)


# Amount

df['amt_diff_1_day_country_post'] = (df.groupby(['accountNumber', 'merchantCountry','posEntryMode','year','month', 'day'])
                                       ['transactionAmount'].diff())


df['num_trans_1_day'] = df.groupby(['accountNumber','year','month', 'day'])['transactionAmount'].transform('count')
df['sum_trans_1_day'] = df.groupby(['accountNumber','year','month', 'day'])['transactionAmount'].transform('sum')
df['num_trans_1_day_country_pos'] = df.groupby(['accountNumber','merchantCountry','posEntryMode','year','month', 'day'])['transactionAmount'].transform('count')
df['sum_trans_1_day_country_pos'] = df.groupby(['accountNumber','merchantCountry','posEntryMode','year','month', 'day'])['transactionAmount'].transform('sum')

# Covert following columns to categorical using pandas .astype


df["accountNumber"] = df["accountNumber"].astype('category')
df["merchantId"] = df["merchantId"].astype('category')
df["merchantZip"] = df["merchantZip"].astype('category')

df["accountNumber"] = df["accountNumber"].cat.codes
df["merchantId"] = df["merchantId"].cat.codes
df["merchantZip"] = df["merchantZip"].cat.codes

# drop transaction time
df = df.drop('transactionTime', axis = 1)
# Fill all NaNs with 0
df.fillna(0, inplace = True)


labels = df.pop("Fraud")

features = df

# Let's first look at the class imbalance
import numpy as np
neg, pos = np.bincount(labels)
total = neg + pos
print('Total transactions: {}\n    Fraudulent: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))


# This shows we are dealing with class imbalance !

# Uncomment to look at the correlation between features

# def corr_check(data):
    
#     plt.style.use('fivethirtyeight')
#     fig, ax = plt.subplots(figsize=(40,35))         # Sample figsize in inches
#     sns.heatmap(data.corr(), linewidths=.5, ax=ax, annot= True)
#     plt.savefig("correlation_heatmap.png", dpi=350, bbox_inches= 'tight')
#     plt.show()

    
# raw_data = pd.concat([features, labels], axis=1, sort=False)
# corr_check(raw_data)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.75, random_state=4, stratify = labels)

classifier_names = ["K-Nearest Neighbors", "Decision Tree", "Random Forest"]
classifiers = [KNeighborsClassifier(n_neighbors=5),  DecisionTreeClassifier(random_state = 7), RandomForestClassifier(random_state=42, max_features='sqrt', 
                            n_estimators= 300, max_depth=10, criterion='entropy')]

ax = plt.gca()
for name, clf in zip(classifier_names, classifiers):
    skf = StratifiedKFold(n_splits=10)
    scores = cross_validate(clf, X_train, y_train, scoring = ['f1', 'precision', 'recall'], cv = skf, n_jobs = 4)

    print("\t------------------------ %s --------------------" %str(name), "\n")
    print("Performance on training set (10-fold stratified cross validation) \n")
    print('Mean precision:%0.4f'%scores['test_precision'].mean(),'\t Mean recall:%0.4f'%scores['test_recall'].mean(),'\t Mean F1-score:%0.4f'%scores['test_f1'].mean(), "\n")

    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Performance of on test set \n")
    print('\tprecision:%0.4f'%prec,'\trecall:%0.4f'%rec,'\tF1-score:%0.4f'%f1, "\n")
    disp = ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    plt.grid(None)
    plt.savefig("Confusion_matrix_" + "%s" %name, dpi=350, bbox_inches= 'tight')
    
    print("\n Classification report \n", classification_report(y_test,y_pred))
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax = ax)
    ax.figure.savefig("ROC.png", dpi=350, bbox_inches= 'tight')
    
    # Get feature importance information

    if name == "Random Forest":
        sorted_idx = clf.feature_importances_.argsort()
        plt.figure()
        plt.rcParams["figure.figsize"] = (15,10)
        plt.barh(X_train.columns[sorted_idx], clf.feature_importances_[sorted_idx])
        plt.savefig("RF_feature_importance.png", dpi=350, bbox_inches= 'tight')
        plt.show()
         
        
    if name == "Decision Tree":
        text_representation = tree.export_text(clf, feature_names = features.columns.values.tolist(), max_depth = 3)
        print(text_representation)

