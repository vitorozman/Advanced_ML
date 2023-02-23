import numpy as np
import pandas as pd
import openml as oml

# Uvoz csv podatkov
dt = pd.read_csv('podatki.csv')

#print(dt.shape)

# odstarnitev prvega stolpca
dt = dt.iloc[:, 1:]
#print(dt)
#print(len(dt.columns))
#print(dt.dtypes)
#print(dt.describe())
#print(dt.isnull().sum())


# Zapolni mankajoce podatke
for c in dt.columns:
    #print(c, dt.isnull().sum()[c])
    if dt.isnull().sum()[c] < len(dt)/5: 
        try:
            mm = dt[c].describe().mean()
        except TypeError:
            mm = dt[c].mode()[0]
        #print(mm)
        dt[c] = dt[c].fillna(mm)
        #print(dt.isnull().sum())
    else:
        dt = dt.drop(columns=[c])
#print(dt.isnull().sum())

#print(dt)

#dt.hist(column='X1')
#print(dt.corr())
##################################################
from sklearn.preprocessing import OneHotEncoder

categ_col = dt.columns[dt.dtypes=='object']
enc = OneHotEncoder(sparse=False)

OH_col = enc.fit_transform(dt[categ_col])

OH_col = pd.DataFrame(OH_col, columns=enc.get_feature_names_out())
OH_col.index = dt.index
num_data = dt.drop(categ_col, axis=1)
data = pd.concat([OH_col, num_data], axis=1)
#print(data)

##################################################
from sklearn.neighbors import KNeighborsClassifier

X = data.drop('Y', axis=1)
Y = data['Y']

knn = KNeighborsClassifier(2)
knn.fit(X, Y)

napoved = knn.predict(X)
acc = sum(napoved == Y)/len(Y)
#print(acc)


##################################################
from sklearn.metrics import accuracy_score
#print(accuracy_score(Y, napoved))


##################################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
KNN = KNeighborsClassifier(3)
KNN.fit(X_train, y_train)
pred = KNN.predict(X_test)
#print(accuracy_score(y_test, pred))

##################################################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xtrain_sc = scaler.fit_transform(X_train)
Xtest_sc = scaler.transform(X_test)


acc = []
k_all = list(range(1, 30))
# Zanka, ki teče po številih sosedov, definiranih v k_all
for k in k_all:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain_sc, y_train)
    pred = knn.predict(Xtest_sc)
    accuracy = accuracy_score(y_test, pred)
    acc += [accuracy]

from matplotlib import pyplot as plt
plt.plot(k_all, acc)
plt.xlabel("stevilo sosedov")
plt.ylabel("natančnost")
#plt.show()


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(Xtrain_sc, y_train)
pred = knn.predict(Xtest_sc)


from sklearn.metrics import confusion_matrix
CMx = confusion_matrix(y_test, pred)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()


from sklearn.metrics import precision_score
preciznost = precision_score(y_test, pred)
#print(preciznost)

from sklearn.metrics import recall_score
priklic = recall_score(y_test, pred)
#print(priklic)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
#print(fpr, tpr, thresholds)
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")
#plt.show()

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, pred)
#print(roc_auc)

from sklearn.linear_model import LinearRegression
data = pd.read_csv('podatki_regresija.csv')
#print(data.head())
X = data.drop('target', axis=1)
Y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
pred = reg.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, pred, squared=False)
#print(rmse)

from sklearn.metrics import r2_score
R2 = r2_score(y_test, pred)
#print(R2)

from sklearn import linear_model
from sklearn.model_selection import cross_validate

lasso = linear_model.Lasso()
cv = cross_validate(lasso, X, Y, cv=5,
    scoring=('r2', 'neg_mean_squared_error'),
    return_train_score=True)

print(cv['test_r2'], cv['test_neg_mean_squared_error'])


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=2, random_state=0)
cvrf = cross_validate(rf, X, Y, cv=5,
    scoring=('r2', 'neg_mean_squared_error'),
    return_train_score=True)

print(cvrf['test_r2'], cvrf['test_neg_mean_squared_error'])


from sklearn import svm

svm = svm.SVC(kernel='linear', C=1, random_state=42)
cvsvm = cross_validate(svm, X, Y, cv=5,
    scoring=('r2', 'neg_mean_squared_error'),
    return_train_score=True)

print(cvsvm['test_r2'], cvsvm['test_neg_mean_squared_error'])