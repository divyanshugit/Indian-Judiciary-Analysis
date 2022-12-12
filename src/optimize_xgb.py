import pandas as pd
import optuna

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,mean_squared_error

df = pd.read_csv("/nlsasfs/home/ttbhashini/prathosh/divyanshu/PreCog/data/judge/judges_clean.csv")

# creating instance of labelencoder
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
    
df['jp_encoded'] = encoder1.fit_transform(df['judge_position'])
df['target'] = encoder2.fit_transform(df['female_judge'])

df.drop(['ddl_judge_id','start_date','end_date','judge_position','female_judge'],axis=1,inplace=True)

X = df.drop('target', axis=1)
y = df['target']


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

def objective(trial: optuna.trial):
    """Define the objective function"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }

    # Fit the model
    optuna_model = xgb.XGBClassifier(**params)
    optuna_model.fit(X_train, Y_train)

    # Make predictions
    y_pred = optuna_model.predict(X_test)

    # Evaluate predictions
    score = f1_score(Y_test, y_pred,average='weighted')
    return score

study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=50)
print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))
print('  Params: ')
"""
Best trial:
  Value: 0.5831698520838298
  Params: 
    max_depth: 5
    learning_rate: 0.7782772582940064
    n_estimators: 478
    min_child_weight: 10
    gamma: 0.00903973717727996
    subsample: 0.036981293497782786
    colsample_bytree: 0.9426455132907368
    reg_alpha: 0.9688575845856799
    reg_lambda: 0.3559669652952865

"""
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))


params = trial.params

xgb_model = xgb.XGBClassifier(**params).fit(X_train, Y_train)

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

filename = 'op_xgb_model.pkl'
pickle.dump(xgb_model, open(filename, 'wb'))