import pandas as pd
import numpy as np
# Visualizations
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
get_ipython().run_line_magic("matplotlib", " inline")
from pandas.plotting import scatter_matrix

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn import svm
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction


##HELPER METHODS
def featuresprocess(df):
    '''
    Use to map features of df 
      usually called for both df_train and df_test to line them up
    '''
    resdf = df.copy()
    resdf = resdf.fillna('Unknown')
    resdf.drop(['gsm'], axis = 1, inplace = True)
    resdf.drop(['page_id'], axis = 1, inplace = True)
    resdf.drop(['urlslug'], axis = 1, inplace = True)
    resdf.drop(['first appearance'], axis = 1, inplace = True)
    resdf.drop(['name'], axis = 1, inplace = True)
#     resdf.drop(['year'],axis = 1, inplace = True)
#     resdf.drop(['id','align'],axis = 1, inplace = True)
#     resdf.drop(['alive'],axis = 1, inplace = True)
#     resdf.drop(['eye','sex', 'hair'],axis = 1, inplace = True)
    return resdf

def split_XYtrain_data(df, test_size=0.2, random_state=0):
  '''Only use for train data (with income>50k label)
  Return Xtrain, Xtest, Ytrain, Ytest
  '''
  X = df.drop(labels=['appearances'],axis=1)
  Y = df['appearances']
  return  train_test_split(X, Y, test_size =test_size, random_state = random_state)

def encode_features(df_train, df_test, onlyone=False):
  '''
  Encode labels for two X data sets
  -If use for split test and train. Pass in Xtrain and Xtest
  -If use for real test and full train. Pass in full X data train, Xtest_kaggle
  '''
  df_combined = pd.concat([df_train, df_test])
  categorical = [column for column in df_combined if df_combined.dtypes[column] == np.object]
  # categorical = ['workclass','education', 'marital.status', 'occupation', 'relationship','race', 'sex']#drop native.country
  for feature in categorical:
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(df_combined[feature])
    df_train[feature] = encoder.transform(df_train[feature])
    if not onlyone:
      df_test[feature] = encoder.transform(df_test[feature])
  return df_train, df_test

def crossvaldata_sklearn(df, split=0.3,rdstate=0):
  """Return preprocessed data, split into train and test
  (not used when apply to kaggle data)
  """
  df = featuresprocess(df)
  Xtrain, Xtest, Ytrain, Ytest = split_XYtrain_data(df, test_size=split, random_state=rdstate)
  Xtrain, Xtest = encode_features(Xtrain, Xtest)
  return Xtrain, Xtest, Ytrain, Ytest



df_train = pd.read_csv('https://raw.githubusercontent.com/baohuy251210/ComicCharactersMining/main/data/wikia/marvel-wikia-data.csv',na_values='?')



print("Missing values:")
df = df_train
total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', 'get_ipython().run_line_magic("'])", "")


##For report only
with open('./train_final_missingvalues.tex', 'w') as tf:
        tf.write(missing_data.to_latex(index=True, float_format="get_ipython().run_line_magic("1.2f",", " label='tab:missingvals',")
                                    caption="Missing values from Train data"))
missing_data.head(5)


total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', 'get_ipython().run_line_magic("'])", "")
missing_data.head(5)
# df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.to_csv("marvel_mostcm.csv", index=False)


# df_train = pd.read_csv("dc_mostcm1.csv", na_values='Unknown')
# df = df_train.dropna(subset=['appearances','year'])
df = pd.read_csv("dc_mostcm2.csv")

scaler = StandardScaler()

from sklearn.metrics import classification_report, mean_squared_error

xtrain, xtest, ytrain, ytest = crossvaldata_sklearn(df,split=0.2,rdstate=2)
xtrain = pd.DataFrame(scaler.fit_transform(xtrain), columns = xtrain.columns)
xtest = pd.DataFrame(scaler.transform(xtest), columns = xtrain.columns)

print(xtrain.head(5))
random_forest =RandomForestRegressor(n_estimators=50, max_depth=3)
random_forest.fit(xtrain, ytrain)
predict_rf = random_forest.predict(xtest)
acc_rf = random_forest.score(xtest, ytest)*100
acc_rf_t = random_forest.score(xtrain, ytrain)*100
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
print(Xkf.head(5))
model = RandomForestRegressor(n_estimators=50, max_depth=3)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print("Sklearn random forest\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(acc_rf_t,acc_rf))
print("MSE on TRAIN: ",mean_squared_error(ytrain, random_forest.predict(xtrain)))
print("MSE on TEST: ",mean_squared_error(ytest, predict_rf))
print('Accuracy: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")





adaboost =AdaBoostRegressor(n_estimators=100)
adaboost.fit(xtrain, ytrain)
predict_ada = adaboost.predict(xtest)
acc_ada = adaboost.score(xtest, ytest)*100
acc_ada_t = adaboost.score(xtrain, ytrain)*100

print("Sklearn Adaboost forest\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(acc_ada_t,acc_ada))
print("MSE on TRAIN: ",mean_squared_error(ytrain, adaboost.predict(xtrain)))
print("MSE on TEST: ",mean_squared_error(ytest, predict_ada))
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
model = AdaBoostRegressor(n_estimators=100)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('Kfold: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")


gdr =GradientBoostingRegressor(n_estimators=1000,max_depth = 3, learning_rate = 0.01)
gdr.fit(xtrain, ytrain)
predict_gdr  = gdr.predict(xtest)
acc_gdr  = gdr.score(xtest, ytest)*100
acc_gdr_t = gdr.score(xtrain, ytrain)*100

print("Sklearn Gradient forest\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(acc_gdr_t,acc_gdr))
print("MSE on TRAIN: ",mean_squared_error(ytrain, gdr.predict(xtrain)))
print("MSE on TEST: ",mean_squared_error(ytest, predict_gdr))
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
model =GradientBoostingRegressor(n_estimators=1000,max_depth = 3, learning_rate = 0.01)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('Kfold: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")


from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
logr = LogisticRegression(max_iter=100)
logr.fit(xtrain, ytrain)
lg_test = logr.score(xtest, ytest)
lg_train = logr.score(xtrain, ytrain)
print("Sklearn Log R\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(lg_test,lg_train))

print("MSE TRAIN: ",mean_squared_error(ytrain, logr.predict(xtrain)))
print("MSE TEST: ",mean_squared_error(ytest, logr.predict(xtest)))
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
model =LogisticRegression(max_iter=100)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('Kfold: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")


linearreg = LinearRegression()
linearreg.fit(xtrain, ytrain)
linearreg_test = linearreg.score(xtest, ytest)
linearreg_train = linearreg.score(xtrain, ytrain)
print("Sklearn Linear Regression\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(linearreg_test,linearreg_train))

print("MSE TRAIN: ",mean_squared_error(ytrain, linearreg.predict(xtrain)))
print("MSE TEST: ",mean_squared_error(ytest, linearreg.predict(xtest)))
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
model =LinearRegression()
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('Kfold: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")


rr = Ridge(alpha=0.1)
rr.fit(xtrain, ytrain)
rr_test = rr.score(xtest, ytest)
rr_train = rr.score(xtrain, ytrain)
print("Sklearn Ridge Regression\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(rr_test,rr_train))

print("MSE TRAIN: ",mean_squared_error(ytrain, rr.predict(xtrain)))
print("MSE TEST: ",mean_squared_error(ytest, rr.predict(xtest)))
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
model =Ridge(alpha=0.1)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('Kfold: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")


lasso = Lasso(alpha=0.01)
lasso.fit(xtrain, ytrain)
lasso_test = lasso.score(xtest, ytest)
lasso_train = lasso.score(xtrain, ytrain)
print("Sklearn Lasso Regression\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(lasso_test,lasso_train))

print("MSE TRAIN: ",mean_squared_error(ytrain, lasso.predict(xtrain)))
print("MSE TEST: ",mean_squared_error(ytest, lasso.predict(xtest)))
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
model =Lasso(alpha=0.01)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('Kfold: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")


from sklearn.linear_model import BayesianRidge
brr = BayesianRidge()
brr.fit(xtrain, ytrain)
brr_test = brr.score(xtest, ytest)
brr_train = brr.score(xtrain, ytrain)
print("Sklearn Bayesian Ridge Regression\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(brr_test,brr_train))

print("MSE TRAIN: ",mean_squared_error(ytrain, brr.predict(xtrain)))
print("MSE TEST: ",mean_squared_error(ytest, brr.predict(xtest)))
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
model =BayesianRidge()
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('Kfold: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=8)
knr.fit(xtrain, ytrain)
knr_test = knr.score(xtest, ytest)
knr_train = knr.score(xtrain, ytrain)
print("Sklearn KNeighborsRegressor\nTrain: {0:5.5f}\nTest: {1:5.5f}".format(knr_test,knr_train))

print("MSE TRAIN: ",mean_squared_error(ytrain, knr.predict(xtrain)))
print("MSE TEST: ",mean_squared_error(ytest, knr.predict(xtest)))
#Kfold
df_kf = featuresprocess(pd.read_csv("dc_mostcm2.csv"))
Xkf = df_kf.drop(['appearances'], axis=1)
Ykf = df_kf['appearances']
Xkf,_ = encode_features(Xkf, Xkf, onlyone=True)

scaler = StandardScaler()
Xkf = pd.DataFrame(scaler.fit_transform(Xkf), columns = Xkf.columns)
model =KNeighborsRegressor(n_neighbors=6)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(model, Xkf, Ykf,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('Kfold: get_ipython().run_line_magic(".3f", " (%.3f)' % (scores.mean(),scores.std()))")






