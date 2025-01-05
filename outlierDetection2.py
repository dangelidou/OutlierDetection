import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Mall_Customers_with_outliers.csv").iloc[:, 3:]  # Use relevant features only

# Step 1: K-Means Clustering for General Clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df)
kmeans_labels = kmeans.labels_
kmeans_centroids = kmeans.cluster_centers_

# Prepare to store results from Hierarchical Clustering
hierarchical_labels = np.zeros_like(kmeans_labels, dtype=int)

# Step 2: Hierarchical Clustering within each K-Means Cluster
current_label = 0  # To ensure unique labels across all sub-clusters
for cluster_id in np.unique(kmeans_labels):
    # Extract data points for the current K-Means cluster
    cluster_data = df[kmeans_labels == cluster_id]
    
    # Apply Hierarchical Clustering
    linkage_matrix = linkage(cluster_data, method='ward')
    sub_cluster_labels = fcluster(linkage_matrix, t=2, criterion='distance')  # t=2 for 2 sub-clusters
    
    # Assign unique labels to hierarchical sub-clusters
    hierarchical_labels[kmeans_labels == cluster_id] = sub_cluster_labels + current_label
    current_label += len(np.unique(sub_cluster_labels))

# Step 3: Outlier Detection
# Compute distances of each point to the centroids of their hierarchical clusters
all_distances = cdist(df, kmeans_centroids, metric='euclidean')
outlier_threshold = np.percentile(all_distances, 95)  # Top 5% as outliers
outliers = np.where(np.min(all_distances, axis=1) > outlier_threshold)[0]

# Visualization
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.7, label='Clusters')
plt.scatter(df.iloc[outliers, 0], df.iloc[outliers, 1], c='red', marker='x', label='Outliers')
plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], marker='*', s=200, c='black', edgecolors='white', label='K-Means Centroids')
plt.title("K-Means and Hierarchical Clustering with Outliers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
