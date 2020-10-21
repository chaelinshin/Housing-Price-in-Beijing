# The following codes are based on codes from course material (Intro to Data Mining, 6103, at the George Washington University)


#%%
# Standard basic checks
def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: dtypes: ')
  cnt+=1
  print(dframe.dtypes)

  try:
    print(f'\n{cnt}: columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
import os
#os.chdir('C:/Users/chael/Documents/Intro-to-Data-Mining-Project)
dirpath = os.getcwd()
filepath = os.path.join( dirpath,'new.csv')
dfhouse = pd.read_csv(filepath, encoding='latin-1')
dfChkBasics(dfhouse)

#%%
# drop irrelevant columns

dfhouse = dfhouse.drop(columns=['url', 'id', 'Lng', 'Lat', 'Cid', 'price', 'floor', 'ladderRatio', 'communityAverage'], axis=1)
dfhouse.head()

# remaining variables: 
#   - tradeTime: the date of transaction
#   - DOM (days on market): total number of days the listing is on the active market (age of a real estate listing)
#   - followers: the number of people following the transaction
#   - totalPrice: total price of house
#   - square: the square (?) of house
#   - livingRoom: number of bedroom
#   - drawingRoom: number of living room
#   - kitchen: number of kitchen
#   - bathRoom: number of bathroom
#   - buildingType: 1 - tower ; 2 - bungalow ; 3 - combination of plate and tower ; 4 - plate
#   - constructionTime: the year of construction
#   - renovationCondition : 1 - including other; 2 - rough ; 3 - simplicity ; 4 - refined decoration 
#   - buildingStructure: 1 - unknown ; 2 - mixed ; 3 - brick and wood ; 4 - brick and concrete ; 5 - steel ; 6 - steel-concrete composite
#   - elevator: 1 - have elevator ; 0 - do not have elevator
#   - fiveYearsProperty: 1 - owner had property for less than 5 years ; 0 - owner had property more than 5 years
#   - subway: 1 - have subway ; 0 - no subway
#   - district: district number (there are a total of 16 districts in Beijing but 3 are excluded in this dataset)

#%%
# rename columns

dfhouse = dfhouse.rename(columns={"totalPrice": "price","tradeTime": "tradedate", "livingRoom": "bedroom", "drawingRoom": "livingroom", "bathRoom": "bathroom", "buildingType": "bldgtype", "constructionTime": "constructyr", "renovationCondition": "condition", "buildingStructure": "structure", "fiveYearsProperty": "fiveyrproperty"})
print(dfhouse.head())
print(dfChkBasics(dfhouse))

#%%
# Pre-processing
# bedroom, livingroom, bathroom, constructyr are all objects
print(dfhouse['bedroom'].unique())
print(dfhouse['livingroom'].unique())
print(dfhouse['bathroom'].unique())
print(dfhouse['constructyr'].unique())

#%%
# for bedroom, change the '#NAME?' value to np.NaN, and change strings to numbers
dfhouse = dfhouse.replace('#NAME?',np.NaN)

try: dfhouse.bedroom = pd.to_numeric( dfhouse.bedroom)
except: print("Cannot handle to_numeric for column: bedroom")
finally: print(dfhouse.bedroom.describe(), '\n', dfhouse.bedroom.value_counts(dropna=False))

print(dfhouse['bedroom'].unique()) # much better

#%%
# clean livingroom variable

def cleanDfliving(row):
  thisroom = row['livingroom']
  if (thisroom == 'ÖÐ 14'): return np.nan
  if (thisroom == 'ÖÐ 15'): return np.nan
  if (thisroom == 'ÖÐ 16'): return np.nan
  if (thisroom == 'ÖÐ 6'): return np.nan
  if (thisroom == '¸ß 14'): return np.nan
  if (thisroom == '¶¥ 6'): return np.nan
  if (thisroom == 'µÍ 6'): return np.nan
  if (thisroom == 'µÍ 16'): return np.nan
  if (thisroom == '¸ß 12'): return np.nan
  if (thisroom == 'µÍ 15'): return np.nan
  if (thisroom == '¸ß 6'): return np.nan
  if (thisroom == 'µ× 28'): return np.nan
  if (thisroom == 'µ× 11'): return np.nan
  if (thisroom == 'ÖÐ 24'): return np.nan
  if (thisroom == 'µ× 20'): return np.nan
  if (thisroom == 'ÖÐ 22'): return np.nan
  
  return thisroom

dfhouse['livingroom'] = dfhouse.apply(cleanDfliving, axis=1) 

try: dfhouse.livingroom = pd.to_numeric( dfhouse.livingroom)
except: print("Cannot handle to_numeric for column: livingroom")
finally: print(dfhouse.livingroom.describe(), '\n', dfhouse.livingroom.value_counts(dropna=False))

print(dfhouse['livingroom'].unique())

#%%
# clean bathroom variable

def cleanDfbath(row):
  thisroom = row['bathroom']
  if (thisroom == 'Î´Öª'): return np.nan
  if (thisroom == 2006): return np.nan
  if (thisroom == 2003): return np.nan
  if (thisroom == 1990): return np.nan
  if (thisroom == 2000): return np.nan
  if (thisroom == 1996): return np.nan
  if (thisroom == 2005): return np.nan
  if (thisroom == 2011): return np.nan
  if (thisroom == 1994): return np.nan
  if (thisroom == '2003'): return np.nan
  if (thisroom == 2004): return np.nan
  
  return thisroom

dfhouse['bathroom'] = dfhouse.apply(cleanDfbath, axis=1) 

try: dfhouse.bathroom = pd.to_numeric( dfhouse.bathroom)
except: print("Cannot handle to_numeric for column: bathroom")
finally: print(dfhouse.bathroom.describe(), '\n', dfhouse.bathroom.value_counts(dropna=False))

print(dfhouse['bathroom'].unique())

#%%
# clean constructyr variable

def cleanDfconstruct(row):
  thisyear = row['constructyr']
  if (thisyear == '1'): return np.nan
  if (thisyear == '0'): return np.nan
  if (thisyear == 'Î´Öª'): return np.nan
  
  return thisyear

dfhouse['constructyr'] = dfhouse.apply(cleanDfconstruct, axis=1) 

try: dfhouse.constructyr = pd.to_numeric( dfhouse.constructyr)
except: print("Cannot handle to_numeric for column: constructyr")
finally: print(dfhouse.constructyr.describe(), '\n', dfhouse.constructyr.value_counts(dropna=False))

print(dfhouse['constructyr'].unique())

#%%
# Remove rows where condition is 0. We don't have information about condition 0.
dfhouse = dfhouse[dfhouse['condition'] > 0 ]
print(dfhouse['condition'].unique())

#%%
# check to see whether data was cleaned properly 
dfChkBasics(dfhouse)

#%%
# correlation matrix 
import seaborn as sns  
plt.figure(figsize=(12,10))
cor = dfhouse.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show
# variables that have relatively strong correlations with price: DOM, square, bedroom, livingroom, bathroom, constructyr, condition

#%%
# histogram of price

price = dfhouse['price']
price = price[ dfhouse['price'] < 2200]

bins = np.linspace(0, 2000, 50)
plt.hist(price.dropna(), bins, label='price',edgecolor='black', color='orange',linewidth=1.2)
plt.xlabel('Housing Price (10,000 yuan)')
plt.ylabel('Frequency')
plt.show()

#%%
# linear regression
from statsmodels.formula.api import ols
modelprice = ols(formula='price ~ square + DOM + bedroom + livingroom + bathroom + C(condition) + C(elevator) + C(subway)', data=dfhouse)
print( type(modelprice))

modelpriceFit = modelprice.fit()
print( type(modelpriceFit) )
print( modelpriceFit.summary() )
# R squared value: 0.461

#%%
# check VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor
X = dfhouse[['square', 'DOM', 'bedroom', 'livingroom', 'bathroom', 'condition', 'elevator', 'subway']].dropna()
X['Intercept'] = 1

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ] 

print(vif)


# Results/Interpretation:
#    Condition
#   - Condition 2 (rough): compared to Condition 1 (including others), price increases 22.72.
#   - condition 3 (simplicity): compared to Condition 1, price increases 72.01.
#   - condition 4 (refined): compared to Condition 1, price increases 85.66.
#   Elevator
#   - Compared to Elevator 0 (no elevator), price increases 41.77.
#   Subway
#   - Compared to Subway 0 (no subway), price increases 89.55.
#   Square
#   - For every 1 unit increase in square, price changes by 3.61.
#   Bathroom
#   - For 1 more bathroom, price changes 22.61.
#   Bedroom
#   - For 1 more bedroom, price changes 22.30.
#   Livingroom
#   - For 1 more livingroom, price changes -8.80.
#   DOM
#   - For 1 unit increase in DOM, price changes 0.80.

# 
#%%
# linear regression with sklearn
# split into train / test sets
from sklearn.model_selection import train_test_split
from sklearn import linear_model

dfhouse2 = dfhouse.dropna()

xhouse = dfhouse2[['square', 'DOM', 'bedroom', 'livingroom', 'bathroom', 'condition', 'elevator', 'subway']]
yhouse = dfhouse2['price']

X_train1, X_test1, y_train1, y_test1 = train_test_split(xhouse, yhouse, test_size = 0.25, random_state=2020)

full_split = linear_model.LinearRegression() 
full_split.fit(X_train1, y_train1) 
y_pred1 = full_split.predict(X_test1)
full_split.score(X_test1, y_test1) 

print('score:', full_split.score(X_test1, y_test1)) 
print('intercept:', full_split.intercept_) 
print('coef_:', full_split.coef_) 
# score: 0.44

#%% 
# Regression Tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as MSE  
from sklearn.preprocessing import scale
xhouse = dfhouse2[['square', 'followers', 'DOM', 'bedroom', 'livingroom', 'condition', 'elevator', 'subway']]
yhouse = dfhouse2['price']

X_train7, X_test7, y_train7, y_test7= train_test_split(xhouse, yhouse, test_size=0.25,random_state=1)

regtree0 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1,random_state=22) 

regtree0.fit(X_train7, y_train7)  
from sklearn.metrics import mean_squared_error as MSE

#%%
# evaluation
y_pred7 = regtree0.predict(X_test7) 
mse_regtree0 = MSE(y_test7, y_pred7)  
rmse_regtree0 = mse_regtree0 ** (.5)
print("Test set RMSE of regtree0: {:.2f}".format(rmse_regtree0))

y_pred8 = regtree0.predict(X_train7)
mse_regtree1 = MSE(y_train7, y_pred8)  
rmse_regtree1 = mse_regtree0 ** (.5)
print("Train set RMSE of regtree1: {:.2f}".format(rmse_regtree1))

# RMSE: 203.76

#%%
from sklearn import linear_model
olshouse = linear_model.LinearRegression() 
olshouse.fit( X_train7, y_train7 )

y_pred_ols = olshouse.predict(X_test7)  

mse_ols = MSE(y_test7, y_pred_ols)  
rmse_ols = mse_ols**(0.5)  

print('Linear Regression test set RMSE: {:.2f}'.format(rmse_ols))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_regtree0))

#%%
# divide price variable into three categories: 'low' (lower than 100), 'medium' (higher than 100 but lower than 750), 'high' (higher than 750)
# cut off our upper limit to 2000 to avoid extreme outliers

bins = [0, 400, 1000, 2000]
names= ['low', 'medium', 'high']
dfhouse['pricelevel'] = pd.cut(dfhouse['price'], bins, labels=names)

# check to see whether the new column was properly included
#dfhouse.head()


#%%
# logistic regression
import statsmodels.api as sm 
from statsmodels.formula.api import glm

# divide price into two categories for logistic regression
bins2 = [0, 400, 2000]
names2 = ['low', 'high']
dfhouse['pricetwolevel'] = pd.cut(dfhouse['price'], bins2, labels=names2)

#%%

modelpricelogitfit = glm(formula='pricetwolevel ~ square  + DOM + bedroom + bathroom + livingroom + C(condition) + C(elevator) + C(subway)', data=dfhouse, family=sm.families.Binomial()).fit()

print(modelpricelogitfit.summary() )

print(np.exp(modelpricelogitfit.params))
print(np.exp(modelpricelogitfit.conf_int()))


# Results/Interpretation
#   Condition
#   - As condition goes up, chance of low price decreases
#   Elevator/Subway
#   - If you have elevator, subway, chance of low price decreases
#   Square
#   - If you have more squares, chance of low price decreases
#   Followers
#   - If you have more followers, chance of low price decreases
#   DOM
#   - If you have more DOM, chance of low price decreases
#   Bedroom/Livingroom
#   - If you have more bedrooms or livingrooms, chance of low price decreases

#%%
modelpredictions = pd.DataFrame( columns=['pricelogitfit'], data= modelpricelogitfit.predict(dfhouse)) 

print(modelpredictions.shape)
print( modelpredictions.head() )

# print(pd.crosstab(dfhouse.pricetwolevel, modelpredictions.pricelogitfit,
#rownames=['Actual'], colnames=['Predicted'],
#margins = True))

#%%
# logistic regression using sklearn
# split into train / test set
dfhouse2 = dfhouse.dropna()
dfhouse2['pricetwonumber'] = dfhouse.pricetwolevel.apply(lambda x: 0 if x == 'low' else 1)
dfhouse2.head()
xhouse = dfhouse2[['square', 'followers', 'DOM', 'bedroom', 'livingroom', 'condition', 'elevator', 'subway']]
yhouse2 = dfhouse2['pricetwonumber']


#%%
# logistic regression using sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train2, X_test2, y_train2, y_test2 = train_test_split(xhouse, yhouse2, test_size = 0.25, random_state=2020)

logreg = LogisticRegression()

logreg.fit(X_train2, y_train2)
y_pred = logreg.predict(X_test2)

# confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test2, y_pred)
cnf_matrix

# accuracy: 0.74
# recall for 'high'(1): 0.58

#%%
# confusion matrix heatmap
class_names = ['low', 'high']
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Price')
plt.xlabel("Predicted Price")
plt.show()


#%%
# Evaluate

from sklearn.metrics import classification_report
print(classification_report(y_test2, y_pred))

print("Accuracy:",metrics.accuracy_score(y_test2,y_pred)) # 0.93
print("Precision score:", metrics.precision_score(y_test2, y_pred, pos_label=1)) # 0.63
print("Recall rate:",metrics.recall_score(y_test2, y_pred, pos_label=1)) # 0.23

#%%
# ROC-AUC for logistic regression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test2, logreg.predict(X_test2))
fpr, tpr, thresholds = roc_curve(y_test2, logreg.predict_proba(X_test2)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.show()

#%%
# KNN
# use pricelevel variable 
# exclude categorical variables - condition, elevator, subway
xhouse = dfhouse2[['square', 'DOM', 'bathroom', 'bedroom', 'livingroom']]
yhouse3 = dfhouse2['pricelevel']

X_train,X_test,y_train,y_test = train_test_split(xhouse,yhouse3,test_size=0.25,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

neighbors = np.arange(1,10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  train_accuracy[i] = knn.score(X_train, y_train)
  test_accuracy[i] = knn.score(X_test, y_test)
  
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
# k=3

#%%
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
knn.score(X_test, y_test) # 0.68

#%%
# confusion matrix
from sklearn.metrics import confusion_matrix
y_predknn = knn.predict(X_test)
matrix = confusion_matrix(y_test,y_predknn)
matrix

#%%
# confusion matrix 2
pd.crosstab(y_test, y_predknn, rownames=['Actual'], colnames=['Predicted'], margins=True)

#%%
# classification report

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predknn))

#%%
# scale variables
k = 3

from sklearn.preprocessing import scale
xshouse = pd.DataFrame( scale(xhouse), columns=xhouse.columns )  
yshouse = yhouse3.copy() 

from sklearn.neighbors import KNeighborsClassifier
knn_scv = KNeighborsClassifier(n_neighbors=k) 

from sklearn.model_selection import cross_val_score
scv_results = cross_val_score(knn_scv, xshouse, yshouse, cv=5)
print(scv_results) 
np.mean(scv_results) # score: 0.64


#%%
# Classification trees
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics

columns = ['square', 'bathroom', 'DOM', 'bedroom', 'livingroom', 'condition', 'elevator', 'subway']
X = dfhouse2[['square', 'bathroom', 'DOM', 'bedroom', 'livingroom', 'condition', 'elevator', 'subway']]
Y = dfhouse2['pricelevel']

X_train5, X_test5, y_train5, y_test5 = train_test_split(X, Y, test_size = 0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train5, y_train5)

y_pred5 = clf.predict(X_test5)

#%%
# Accuracy

print("Accuracy:",metrics.accuracy_score(y_test5, y_pred5)) # 0.70
# confusion matrix
print(pd.crosstab(y_test5, y_pred5, rownames=['Actual'], colnames=['Predicted'], margins=True))

from sklearn.metrics import classification_report
print(classification_report(y_test5, y_pred5))

#%%
# ! pip install graphviz
# ! pip install pydotplus

#%%
# gini, entropy

clf2 = DecisionTreeClassifier(criterion="gini")
clf2=clf2.fit(X_train5, y_train5)
y_pred6 = clf2.predict(X_test5)
print('Criterion=gini Accuracy: ', metrics.accuracy_score(y_test5, y_pred6))

clf3 = DecisionTreeClassifier(criterion="entropy")
clf3=clf3.fit(X_train5, y_train5)
y_pred7 = clf3.predict(X_test5)
print('Criterion=entropy Accuracy: ', metrics.accuracy_score(y_test5, y_pred7))

#%%
max_depth = []
acc_gini = []
acc_entropy = []
for i in range(1,30):
 clf2 = DecisionTreeClassifier(criterion='gini', max_depth=i)
 clf2.fit(X_train5, y_train5)
 y_pred6 = clf2.predict(X_test5)
 acc_gini.append(metrics.accuracy_score(y_test5, y_pred6))
 ####
 clf3 = DecisionTreeClassifier(criterion='entropy', max_depth=i)
 clf3.fit(X_train5, y_train5)
 y_pred7 = clf3.predict(X_test5)
 acc_entropy.append(metrics.accuracy_score(y_test5, y_pred7))
 ####
 max_depth.append(i)
d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
 'acc_entropy':pd.Series(acc_entropy),
 'max_depth':pd.Series(max_depth)})
# visualizing changes in parameters
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()

#%%
clf4 = DecisionTreeClassifier(criterion='entropy', max_depth=9)
clf4.fit(X_train5, y_train5)
y_pred8 = clf4.predict(X_test5)
print('Criterion=entropy, Max_depth=9 Accuracy: ', metrics.accuracy_score(y_test5, y_pred8))


#%%
#import os

#os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import pydotplus
#dot_data = StringIO()
#export_graphviz(clf2, out_file=dot_data,  
#                filled=True, rounded=True,
 #               special_characters=True, feature_names = columns, class_names=['low','high'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('price.png')
#Image(graph.create_png())


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

#%%
# logistic regression
lr = LogisticRegression()
lr.fit(X_train2,y_train2)
print(f'lr train score:  {lr.score(X_train2,y_train2)}') # 0.93
print(f'lr test score:  {lr.score(X_test2,y_test2)}') # 0.93
print(confusion_matrix(y_test2, lr.predict(X_test2))) # confusion matrix
print(classification_report(y_test2, lr.predict(X_test2))) # classification report

print("\nReady to continue.")

#%%
# SVC
svc = SVC()
svc.fit(X_train2,y_train2)
print(f'svc train score:  {svc.score(X_train2,y_train2)}') # 0.93
print(f'svc test score:  {svc.score(X_test2,y_test2)}') # 0.93
print(confusion_matrix(y_test2, svc.predict(X_test2))) # confusion matrix
print(classification_report(y_test2, svc.predict(X_test2))) # classification report

print("\nReady to continue.")

#%%
# SVC(kernel="linear")

svc2 = SVC(kernel="linear")
svc2.fit(X_train2,y_train2)
print(f'svc2 train score:  {svc2.score(X_train2,y_train2)}') # 
print(f'svc2 test score:  {svc2.score(X_test2,y_test2)}') # 
print(confusion_matrix(y_test2, svc2.predict(X_test2))) # confusion matrix
print(classification_report(y_test2, svc2.predict(X_test2))) # classification report

print("\nReady to continue.")

#%%
# LinearSVC

svc3 = LinearSVC()
svc3.fit(X_train2,y_train2)
print(f'svc3 train score:  {svc3.score(X_train2,y_train2)}') # 0.36
print(f'svc3 test score:  {svc3.score(X_test2,y_test2)}') # 0.36
print(confusion_matrix(y_test2, svc3.predict(X_test2))) # confusion matrix
print(classification_report(y_test2, svc3.predict(X_test2))) # classification report

print("\nReady to continue.")

#%%
# KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train2,y_train2)
print(f'knn train score:  {knn.score(X_train2,y_train2)}') # 0.95
print(f'knn test score:  {knn.score(X_test2,y_test2)}') # 0.92
print(confusion_matrix(y_test2, knn.predict(X_test2))) # confusion matrix
print(classification_report(y_test2, knn.predict(X_test2))) # classification report

print("\nReady to continue.")

#%%
# DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train2,y_train2)
print(f'tree train score:  {tree.score(X_train2,y_train2)}') # 1.0
print(f'tree test score:  {tree.score(X_test2,y_test2)}') # 0.91
print(confusion_matrix(y_test2, tree.predict(X_test2))) # confusion matrix
print(classification_report(y_test2, tree.predict(X_test2))) # classification report

print("\nReady to continue.")


#%%


