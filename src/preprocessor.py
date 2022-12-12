import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("PreCog/data/judge/judges_clean.csv")
    
# creating instance of labelencoder
encoder = LabelEncoder()
    
df['label'] = encoder.fit_transform(df['female_judge'])

state_df = pd.read_csv("PreCog/data/keys/cases_state_key.csv")
dist_df = pd.read_csv("PreCog/data/keys/cases_district_key.csv")
court_df = pd.read_csv("PreCog/data/keys/cases_court_key.csv")

state_mapper = dict(zip(state_df.state_code,state_df.state_name))
dist_mapper = dict(zip(dist_df.dist_code,dist_df.district_name))

df["State"] = df.state_code.map(state_mapper)
df["District"] = df.dist_code.map(dist_mapper)

data = df.drop(["ddl_judge_id","state_code", "dist_code","start_date","end_date"],axis=1)

data['inputs'] = data['judge_position'] +', '+ data['State'] + ", " + data['District']
print(data.info())

"""
>>> print(data.info())

Data columns (total 7 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   court_no        98478 non-null  int64 
 1   judge_position  98478 non-null  object
 2   female_judge    98477 non-null  object
 3   label           98478 non-null  int64 
 4   State           98478 non-null  object
 5   District        98478 non-null  object
 6   inputs          98478 non-null  object
dtypes: int64(2), object(5)
"""

data = data.drop(['court_no','judge_position','State','District'],axis=1)

"""
>>> print(data.info())

 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   female_judge  98477 non-null  object
 1   label         98478 non-null  int64 
 2   inputs        98478 non-null  object
dtypes: int64(1), object(2)
"""

train_df, val_df = train_test_split(data, test_size=0.25,shuffle=False)
train_df.to_csv('train_e.csv',index=False)
val_df.to_csv('val_e.csv',index=False)
