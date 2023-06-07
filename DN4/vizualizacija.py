import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




eq1 = pd.read_csv('DN4/Rezultati/rezultati1.csv')

eq1 = np.array(eq1.sort_values('Prob_algo', ascending=False).drop('iter',axis=1))
eq1 = eq1[[4, 5, 6, 7, 10, 12, 14, 16, 17, 18]]
eq1 = eq1[:, 1:3]

eq2 = pd.read_csv('DN4/Rezultati/rezultati2.csv')
eq2 = np.array(eq2.sort_values('Prob_algo', ascending=False).drop('iter',axis=1))


eq3 = pd.read_csv('DN4/Rezultati/rezultati3.csv')
eq3 = np.array(eq3.sort_values('Prob_algo', ascending=False).drop('iter',axis=1))


# Data
x = np.transpose(np.array([range(100, 1001, 100)]))

a1 = eq1[:,0]
b1 = eq1[:,1]
a2 = eq2[:,1]
b2 = eq2[:,2]
a3 = eq3[:,1]
b3 = eq3[:,2]


plt.plot(x, a1,  color='darkred', label='EQ1_algo')
plt.plot(x, b1,  color='lightcoral', label='EQ1')
plt.xlim(100, 1000)
plt.ylim(0, 15)
plt.xlabel('Število enačb')
plt.ylabel('Napaka')
plt.legend()
plt.savefig('DN4/Rezultati/eq1.png')
plt.clf()

plt.plot(x, a2,  color='darkgreen', label='EQ2_algo')
plt.plot(x, b2,  color='lightgreen', label='EQ2')
plt.xlim(100, 1000)
plt.ylim(-0.5, 15)
plt.xlabel('Število enačb')
plt.ylabel('Napaka')
plt.legend()
plt.savefig('DN4/Rezultati/eq2.png')
plt.clf()

plt.plot(x, a3,  color='darkblue', label='EQ3_algo')
plt.plot(x, b3,  color='lightblue', label='EQ3')
plt.xlim(100, 1000)
plt.ylim(0, 1)
plt.xlabel('Število enačb')
plt.ylabel('Napaka')
plt.legend()
plt.savefig('DN4/Rezultati/eq3.png')

