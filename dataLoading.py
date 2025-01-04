import pandas as pd
import os
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


file_name = 'Mall_Customers.csv'
df = pd.read_csv(file_name)

# kept the last two columns which are the ones that define the clusters
df = df.iloc[:, 3:]

# pca = PCA(n_components=2, svd_solver='auto')
# data_reduced = pca.fit_transform(df_scaled)

#calculating silhouette score to find the optimal number of clusters
silhouette_scores = []
for k in range(2, 20):  # Test for a range of k values
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df)
    score = silhouette_score(df, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 20), silhouette_scores, marker='o')
plt.title("Silhouette Scores for K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()


kmeans = KMeans(n_clusters=5).fit(df)
print(kmeans.score(df))

# Plot the reduced data
# plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=kmeans.labels_)

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=kmeans.labels_)
plt.title("5-Means Clustering of Mall Customers")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.show()