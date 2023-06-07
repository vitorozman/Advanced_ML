
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


###################################################################################
# Linear regrestion
###################################################################################

def linearna_regresija(X, y, meja=1e-2):
    imena = X.columns
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    izraz = ""
    pred = np.zeros(len(X))
    for i,b in enumerate(beta):
        if abs(b) > meja:
            if len(izraz) > 0:
                izraz += " + "
            izraz +=  f"{b:.3f}*{imena[i]}"
            pred = pred + b*X.iloc[:, i]
    err = mean_squared_error(y, pred)
    return izraz, err

###################################################################################
# Ridge regrestion
###################################################################################

def ridge_regresija(X, y, lam=1, meja=1e-2):
    imena = X.columns
    beta = np.linalg.pinv(X.T.dot(X) + lam*np.identity(X.shape[1])).dot(X.T).dot(y)
    izraz = ""
    pred = np.zeros(len(X))
    for i,b in enumerate(beta):
        if abs(b) > meja:
            if len(izraz) > 0:
                izraz += " + "
            izraz +=  f"{b:.3f}*{imena[i]}"
        pred = pred + b*X.iloc[:, i]
    err = mean_squared_error(y, pred)
    return izraz, err


###################################################################################
# Lasso regrestion
###################################################################################

from scipy.optimize import minimize

def lasso_regresija(X, y, lam=1, meja=1e-2):
    imena = X.columns

    def f(beta):
        yhat = X.dot(beta)
        return np.sum((yhat-y)**2) + lam*np.sum(np.abs(beta))
    
    beta = minimize(f, np.random.random(X.shape[1]))["x"]
    izraz = ""
    pred = np.zeros(len(X))
    for i,b in enumerate(beta):
        if abs(b) > meja:
            if len(izraz) > 0:
                izraz += " + "
            izraz +=  f"{b:.3f}*{imena[i]}"
        pred = pred + b*X.iloc[:, i]
    err = mean_squared_error(y, pred)
    return izraz, err


###################################################################################
# Bacon 
###################################################################################

def bacon(df, max_iter=20):
    for iteracija in range(max_iter):
        stolpci = list(df.columns)
        df_array = np.array(df)
        # ocenimo, ali obstaja konstanta
        sig = np.std(df_array, axis=0)
        # testirajmo se trivialnost
        if min(sig) < 10 ** -12:  # pazimo na skalo?
            break
        # izracunajmo vrstne rede, najdemo korelacije med njimi
        vrstni_redi = np.array(df.rank(axis=0)).T
        korelacije = np.corrcoef(vrstni_redi)  # korelacije[i, j]
        n = korelacije.shape[0]
        korelacije = [(abs(korelacije[i, j]), (i, j), korelacije[i, j] < 0) for i in range(n) for j in range(i + 1, n)]
        korelacije.sort(reverse=True)
        for kakovost, (i, j), je_mnozenje in korelacije:
            if je_mnozenje:
                ime_novega = f"({stolpci[i]}) * ({stolpci[j]})"
                vrednosti_novega = df_array[:, i] * df_array[:, j]
            else:
                ime_novega = f"({stolpci[i]}) / ({stolpci[j]})"
                vrednosti_novega = df_array[:, i] / df_array[:, j]
            if ime_novega not in stolpci:
                df[ime_novega] = vrednosti_novega
                break
    # najdemo "konstanto"
    df_array = np.array(df)
    sig = np.std(df_array, axis=0)
    i = np.argmin(sig)
    const = np.mean(df_array[:, i])
    print(f"{const:.5e} = {df.columns[i]} (napaka: {sig[i]})")



###################################################################################
# Helpfull function
###################################################################################


def XY(data):
    X = data.drop('Q', axis=1)
    y = data['Q']
    return X, y