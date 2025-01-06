import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# This code was created by chatGPT using this prompt
# 
# "In this dataset, how can we combine the kmeans algorithm with a 
# hierarchical algorithm one after the other in that order, in order to find the outliers?"
# 
# --------------------------------------------------
# Note: This code does not solve the problem in the way we need. We added it for comparison purposes.

# Load dataset
df = pd.read_csv("Mall_Customers_with_outliers.csv").iloc[:, 3:]  # Use relevant features only

# Step 1: K-Means Clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 2: Hierarchical Clustering on K-Means Centroids
hierarchical_model = linkage(centroids, method='ward')
hierarchical_labels = fcluster(hierarchical_model, t=2, criterion='maxclust')

# Step 3: Identify Outliers
distances = cdist(df, centroids, 'euclidean')
min_distances = np.min(distances, axis=1)
outlier_threshold = np.percentile(min_distances, 95)  # Top 5% distances are outliers
outliers = np.where(min_distances > outlier_threshold)[0]

# Visualization
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis', alpha=0.7, label='Normal Points')
plt.scatter(df.iloc[outliers, 0], df.iloc[outliers, 1], c='red', label='Outliers', marker='x')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', edgecolors='white', label='Centroids')
plt.title("K-Means with Hierarchical Outlier Detection")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
