import pandas as pd
import plotly.express as px

df = pd.read_csv("star.csv")
print (df.head())

from sklearn.cluster import KMeans

X = df.iloc[:, [0, 1]].values

print(X)

wcss = []
for i in range(1, 11):
    kmeans=KMeans(n_cluster=i, init='k-means++', random_state = 42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

import seaborn as sns
import matplotlib_pyplot as plt

plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == (0, 0)], X[y_kmeans == (0, 1)], color='#00FFFF', lable = 'coustomer 1')
sns.scatterplot(X[y_kmeans == (1, 0)], X[y_kmeans == (1, 1)], color='#ff0000', lable = 'coustomer 2')
sns.scatterplot(X[y_kmeans == (2, 0)], X[y_kmeans == (2, 1)], color='#00ff33', lable = 'coustomer 3')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='#ddff00', label = 'cluster', s=100, marker=',')
plt.grid(False)
plt.title('cluster of interstellar object')
plt.xlable('Size')
plt.ylable('light')
plt.legend()
plt.show()
