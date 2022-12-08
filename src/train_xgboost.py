import numpy as np
import pandas as pd


from scipy import stats
from sklearn.metrics import r2_score,mean_absolute_error
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score,mean_squared_error


df = pd.read_csv("divyanshu/PreCog/data/judge/judges_clean.csv")
from sklearn.preprocessing import LabelEncoder
    
# creating instance of labelencoder
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
    
df['jp_encoded'] = encoder1.fit_transform(df['judge_position'])
df['target'] = encoder2.fit_transform(df['female_judge'])

df.drop(['ddl_judge_id','start_date','end_date','judge_position','female_judge'],axis=1,inplace=True)

X = df.drop('target', axis=1)
y = df['target']

from sklearn.model_selection import train_test_split
X_train,X_comb,Y_train,Y_comb = train_test_split(X,y,test_size=0.3,random_state=0 , shuffle = False)
X_validation,X_test,Y_validation,Y_test = train_test_split(X_comb,Y_comb,test_size=0.5,random_state=0 , shuffle = False)

print("Shape:",X_train.shape,Y_train.shape)


xgb_model = xgb.XGBClassifier().fit(X_train, Y_train)

y_pred_val = xgb_model.predict(X_validation)
y_pred_test = xgb_model.predict(X_test)

print('Accuracy of XGB regressor on training set: {:.2f}'
       .format(xgb_model.score(X_train, Y_train)))
print('Accuracy of XGB regressor on validation set: {:.2f}'
       .format(xgb_model.score(X_validation, Y_validation)))
print('Accuracy of XGB regressor on test set: {:.2f}'
       .format(xgb_model.score(X_test, Y_test)))
print("f1 Score on validation set: {:.2f}"
       .format(f1_score(Y_validation, y_pred_val, average='weighted')))
print("f1 Score on test set: {:.2f}"
       .format(f1_score(Y_test, y_pred_test, average='weighted')))
print("Mean Squared Error on test: {:.2f}"
       .format(mean_squared_error(Y_test, y_pred_test, squared=False)))
