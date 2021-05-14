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



# print(data.head())
# plt.scatter(data['Age'],data['Annual_Income_(k$)'])
# train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
# df1 = pd.DataFrame({'CustomerID': test_set['CustomerID'],
#                     'Genre': test_set['Genre'],
#                     'Age': test_set['Age'],
#                     'x': test_set['x'],
#                     'Spending_Score': test_set['Spending_Score']})
# print(df1.describe())
# plt.show()
# data['spending_score_cat'] = pd.cut(data['Spending_Score'],
#                                     bins=[0, 20, 40, 60, np.inf],
#                                     labels=[1, 2, 3, 4])
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(data, data['spending_score_cat']):
#     strat_train_set = data.loc[train_index]
#     strat_test_set = data.loc[test_index]
#     print(strat_test_set["spending_score_cat"].value_counts())

# strat_test_set['spending_score_cat']
# print(strat_test_set)
# print(strat_train_set)