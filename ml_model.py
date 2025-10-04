import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

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


'''
Based on the Research article: 
Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification.Page 5/6
Work of: Luz, T. S. F., Braga, R. A. S., & Ribeiro, E. R. (2024). 
Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification. Electronics, 13(19), 3950. 
https://doi.org/10.3390/electronics13193950
'''

                                        #1-Dropping collumns(3 Parts)
df.drop(['kepid','kepoi_name','kepler_name','koi_pdisposition','koi_score','koi_teq_err1','koi_teq_err2'], axis=1, inplace=True) #1st Drop

df.drop(['koi_tce_delivname','ra','koi_time0bk','dec'], axis=1, inplace=True) # 2nd Drop

df.drop(columns=['koi_period_err1','koi_period_err2','koi_duration_err1','koi_duration_err2',
                 'koi_depth_err1','koi_depth_err2','koi_prad_err1','koi_prad_err2',
                 'koi_insol_err1','koi_insol_err2','koi_impact_err1','koi_impact_err2',
                 'koi_steff_err1','koi_steff_err2','koi_slogg_err1','koi_slogg_err2',
                 'koi_srad_err1','koi_srad_err2'],axis=1, inplace=True) # 3rd Drop


                                        #3-Dropping False Positive rows
df= df[df["koi_disposition"] != "FALSE POSITIVE"] 

                                        #4-Mapping Confirmed/Candidate to Binary
mapping = {'CONFIRMED': '0', 'CANDIDATE': '1'}
df['koi_disposition'] = df['koi_disposition'].replace(mapping)

                                        #5-Getting X/Y Features
X = df.drop(columns=['koi_disposition']).values
Y = df['koi_disposition'].values

                                       #6-Setting Training/Testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50, shuffle=True)



                                       #7-PipeLine 
imputer = SimpleImputer(strategy='mean')
pipeline = Pipeline([
    ('imputer', imputer),  #Seems like there are some other NaNs
    ('scaler', StandardScaler()),
    ('adaboost', AdaBoostClassifier(n_estimators=974, learning_rate=0.11, random_state=50))
])

                                        #9-Training 
#Train actual Pipeline
pipeline.fit(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print("Test set accuracy:", test_accuracy)

                                        #10-Saving Model
# Save Model
joblib.dump(pipeline, 'model.pkl')
# Later: load it back
model = joblib.load('model.pkl') #Saved as model.pkl , we will use this model 