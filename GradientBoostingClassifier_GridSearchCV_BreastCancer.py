import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv(r"D:\CDAC ML\Cases\Wisconsin\BreastCancer.csv")
dum_df = pd.get_dummies(df,drop_first=True)

X = dum_df.iloc[:,1:-1]
y = dum_df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2022,test_size=0.3,stratify=y)

model = GradientBoostingClassifier(random_state=2022)
model.fit(X_train,y_train)
y_pred_prob = model.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))


#################################
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
model = GradientBoostingClassifier(random_state=2022)
params = {'learning_rate':np.linspace(0.001,1,5),
          'n_estimators': [100,150],
          'max_depth': [2,3,5]}
gcv = GridSearchCV(model,scoring='roc_auc',cv=kfold,param_grid=params)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)