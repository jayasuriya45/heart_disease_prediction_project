import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pickle 
import warnings
warnings.filterwarnings('ignore')
data_1=pd.read_csv('values.csv')
data_2=pd.read_csv('labels.csv')
data_2=data_2.iloc[:,-1]
combined_data=pd.concat([data_1,data_2],axis=1)
combined_data.drop('patient_id',axis=1,inplace=True)
# Feature engineering


le=OrdinalEncoder()
combined_data['thal']=le.fit_transform(combined_data[['thal']])
x=combined_data.iloc[:,:-1]
y=combined_data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=27,stratify=y)


xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train, y_train)
# y_pred_xgb = xgb_model.predict(x_test)

with open (r"xgb_model.pkl","wb") as pickle_file:
    pickle.dump(xgb_model,pickle_file)
