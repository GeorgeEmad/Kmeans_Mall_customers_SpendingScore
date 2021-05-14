import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import seaborn as sns

from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("Mall_Customers.csv")
data.drop_duplicates(inplace= True)
x = data.iloc[:, [3,4]].values
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
# sns.lineplot(x = range(1, 11),y = wcss, color = 'red')
# plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(x)
for i in range(5):
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], label = "cluster" + str(i+1))
    plt.legend()
plt.grid(False)

plt.show()


