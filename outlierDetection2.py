import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt

# Reload the dataset to ensure correct path
df = pd.read_csv("Mall_Customers_with_outliers.csv").iloc[:, 3:]  # Use relevant features only

# Step 1: K-Means Clustering for General Clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df)
kmeans_labels = kmeans.labels_
kmeans_centroids = kmeans.cluster_centers_

# Prepare for Hierarchical Clustering results and visualization
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
fig.suptitle("Dendrograms for Hierarchical Clustering within K-Means Clusters", fontsize=16)

current_label = 0  # To ensure unique labels across all sub-clusters
hierarchical_labels = np.zeros_like(kmeans_labels, dtype=int)

for cluster_id in range(5):  # Iterate over K-Means clusters
    cluster_data = df[kmeans_labels == cluster_id]  # Data points in the cluster
    
    if len(cluster_data) > 1:  # Hierarchical clustering requires more than one point
        # Apply Hierarchical Clustering
        linkage_matrix = linkage(cluster_data, method='single')
        
        # Calculate dynamic distance threshold (e.g., 50% of max distance in the linkage matrix)
        max_distance = np.max(linkage_matrix[:, 2])
        dynamic_t = 0.5 * max_distance  # Adjust fraction as needed
        
        # Use dynamic threshold for fcluster
        sub_cluster_labels = fcluster(linkage_matrix, t=dynamic_t, criterion='distance')
        
        # Plot dendrogram with y-axis labels for distances
        dendrogram(linkage_matrix, ax=axes[cluster_id], truncate_mode='lastp', p=10, color_threshold=0)
        axes[cluster_id].set_title(f"Cluster {cluster_id + 1}")
        axes[cluster_id].set_xlabel("Points")
        axes[cluster_id].set_ylabel("Distance")
        
        # Ensure y-axis labels are displayed correctly
        axes[cluster_id].tick_params(axis='y', which='both', left=True, labelleft=True)

        # Assign unique labels to hierarchical sub-clusters
        hierarchical_labels[kmeans_labels == cluster_id] = sub_cluster_labels + current_label
        current_label += len(np.unique(sub_cluster_labels))
    else:
        axes[cluster_id].set_title(f"Cluster {cluster_id + 1} (Single Point)")
        axes[cluster_id].axis('off')  # Hide axes for clusters with a single point

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for title
plt.show()

# Step 3: Updated Outlier Detection
outlier_clusters = []
point_outliers = []

for cluster_id in range(5):  # Iterate over K-Means clusters
    cluster_data = df[kmeans_labels == cluster_id]  # Data points in the cluster
    cluster_center = kmeans_centroids[cluster_id]  # Center of the K-Means cluster
    
    # Get subcluster labels for this cluster
    subcluster_ids = np.unique(hierarchical_labels[kmeans_labels == cluster_id])
    
    for subcluster_id in subcluster_ids:
        subcluster_data = cluster_data[hierarchical_labels[kmeans_labels == cluster_id] == subcluster_id]
        subcluster_center = subcluster_data.mean(axis=0)  # Center of the subcluster
        
        # Condition 1: Subcluster as Outlier Cluster
        subcluster_to_cluster_dist = euclidean(subcluster_center, cluster_center)
        cluster_std_dev = np.std(cluster_data, axis=0).mean()  # Avg std dev across features
        
        if subcluster_to_cluster_dist > 2 * cluster_std_dev:
            outlier_clusters.append(subcluster_id)
            continue  # Skip to next subcluster
        
        # Condition 2: Points within Subcluster as Outliers
        distances_to_subcluster_center = np.linalg.norm(subcluster_data - subcluster_center, axis=1)
        subcluster_std_dev = np.std(distances_to_subcluster_center)
        
        point_outliers.extend(
            subcluster_data[distances_to_subcluster_center > 2 * subcluster_std_dev].index
        )

# Visualization
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.7, label='Clusters')
plt.scatter(df.iloc[point_outliers, 0], df.iloc[point_outliers, 1], c='red', marker='x', label='Point Outliers')
plt.scatter(
    kmeans_centroids[:, 0], kmeans_centroids[:, 1], 
    marker='*', s=200, c='black', edgecolors='white', label='K-Means Centroids'
)
for subcluster_id in outlier_clusters:
    subcluster_data = df[hierarchical_labels == subcluster_id]
    plt.scatter(subcluster_data.iloc[:, 0], subcluster_data.iloc[:, 1], edgecolors='red', facecolors='none', s=100, label=f'Outlier Subcluster {subcluster_id}')
    
plt.title("Outliers and Outlier Subclusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
