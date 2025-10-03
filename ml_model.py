import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold

url = "https://raw.githubusercontent.com/MatiasPF1/KOI-Dataset/main/NasaDataset.csv"
df = pd.read_csv(url)
pd.set_option('display.max_columns', None)  # show all columns

#df.shape (9564, 49) 9564 training samples, 49 collumns of features

'''
Based on the Research article: 
Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification.Page 5/6
Work of: Luz, T. S. F., Braga, R. A. S., & Ribeiro, E. R. (2024). 
Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification. Electronics, 13(19), 3950. 
https://doi.org/10.3390/electronics13193950
'''

#Dropping collumns
df.drop(['kepid','kepoi_name','kepler_name','koi_pdisposition','koi_score','koi_teq_err1','koi_teq_err2'], axis=1, inplace=True)
'''
drop(): will drop collumns
      Arguments
      1-['']: whatever collumn we want to drop
      2-axis: x(0) or y(1) axis that want to be dropped
      3-inplace modifies original dataFrame
'''
df.head(5)
#df.shape (9564, 42)
#Exploring koi_tce_delivname
df['koi_tce_delivname'].value_counts()  #Getting how many categories and frequency
df['koi_tce_delivname'].isna().sum() #sum of Nan values
df = df.dropna(subset=['koi_tce_delivname']) #Dropping rows where the Koi_tce_delivname has nan values

#df.shape 334 rows eliminated, all clean
#df['koi_disposition'].value_counts()  Getting how many categories and frequency
df= df[df["koi_disposition"] != "FALSE POSITIVE"] #Dropping FALSE POSITIVE rows
#df['koi_disposition'].value_counts() confirmed 2735 candidate 1919

mapping = {'CONFIRMED': '0', 'CANDIDATE': '1'}
df['koi_disposition'] = df['koi_disposition'].replace(mapping)

#1- Getting X/Y Features
X = df.drop(columns=['koi_disposition']).values
Y = df['koi_disposition'].values

imputer = SimpleImputer(strategy='mean')
pipeline = Pipeline([
    ('imputer', imputer),  #Seems like there are some other NaNs
    ('scaler', StandardScaler()),
    ('adaboost', AdaBoostClassifier(n_estimators=974, learning_rate=0.11, random_state=50))
])

# Stratified 10-fold cross-validation - Testing how the Model would probably gonna work in an last performance
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
print("Cross-validation accuracy for each fold:", cv_scores)
print("Mean cross-validation accuracy:", np.mean(cv_scores))

#Train actual Pipeline
pipeline.fit(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print("Test set accuracy:", test_accuracy)