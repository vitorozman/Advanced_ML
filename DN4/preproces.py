
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)

from sklearn.preprocessing import PolynomialFeatures

def save_all_datas(df, k):
    X = df.drop('Q', axis=1)
    y = df['Q']
    # generate more param.
    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)
    novi_cleni = [el.replace(" ", "•") for el in poly.get_feature_names_out()]
    data_ext = pd.DataFrame(X, columns=novi_cleni)
    # save data
    data_ext['Q'] = y
    data_ext.to_csv(f"NEW_COLL_DATA_ALL{k}.csv")
    data_extt = data_ext.drop(['Tw', 'Ta', 'theta', 'eta'], axis=1)
    data_extt.to_csv(f"NEW_COLL_DATA_{k}.csv")
    print(f"Data {k} successful saved!")


data = pd.read_csv('DN4_1_podatki.csv')
data_all = data
#print(data_all.describe())

data1 = data_all
# Use assuptions about data
data1['sin(theta)'] = np.sin(data1['theta'])
data1['(Tw-Ta)'] = data1['Tw'] - data1['Ta']
data1['(Tw-Ta)^(1/2)'] = (data1['Tw'] - data1['Ta'])**0.5
data1['1/eta'] = (1 / data1['eta'])

data2 = data_all
data2['1-cos(theta)'] = 1-np.cos(data2['theta'])
data2['(Tw-Ta)'] = data2['Tw'] - data2['Ta']
data2['(Tw-Ta)^(1/2)'] = (data2['Tw'] - data2['Ta'])**0.5
data2['1/eta'] = (1 / data2['eta'])

data3 = data_all
data3['sin(theta)'] = np.sin(data3['theta'])
data3['cos(theta)'] = np.cos(data3['theta'])
data3['(Tw-Ta)'] = data3['Tw'] - data3['Ta']
data3['(Tw-Ta)^(1/2)'] = (data3['Tw'] - data3['Ta'])**0.5
data3['1/eta'] = (1 / data3['eta'])


data4 = data_all
data4['sin(theta)'] = np.sin(data4['theta'])
data4['cos(theta)'] = np.cos(data4['theta'])
data4['(Tw-Ta)'] = data4['Tw'] - data4['Ta']
data4['(Tw-Ta)^(1/2)'] = (data4['Tw'] - data4['Ta'])**0.5
data4['-eta'] = (-data4['eta'])

save_all_datas(data1, 1)
save_all_datas(data2, 2)
save_all_datas(data3, 3)
save_all_datas(data4, 4)
