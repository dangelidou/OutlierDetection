import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import silhouette_score
import seaborn as sns
        

def distance_matrix_visualization(df, hierarchical_labels):
    # Create a 2D distance heatmap
    pairwise_distances = pdist(df)
    distance_matrix = squareform(pairwise_distances)

    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap="viridis", square=True, cbar=True)
    plt.title("2D Heatmap of Pairwise Distances")
    plt.xlabel("Points")
    plt.ylabel("Points")
    plt.show()

    # Visualize hierarchical cluster assignments
    plt.figure(figsize=(10, 8))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=hierarchical_labels, cmap='tab20', alpha=0.7)
    plt.scatter(
        kmeans_centroids[:, 0], kmeans_centroids[:, 1],
        marker='*', s=200, c='red', edgecolors='black', linewidths=1.5
    )
    plt.title("Hierarchical Cluster Assignments (Last Iteration)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_inertia(data, max_clusters=10):
    # Calculate the inertia for a range of cluster counts
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Plot the inertia values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Inertia Plot for K-Means Clustering")
    plt.grid(True)
    plt.show()

def plot_silhouette(data, max_clusters=10):
    # Calculate the silhouette score for a range of cluster counts
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, kmeans_labels))

    # Plot the silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Plot for K-Means Clustering")
    plt.grid(True)
    plt.show()

# Reload the dataset to ensure correct path
df = pd.read_csv("Mall_Customers.csv").iloc[:, 3:]  # Use relevant features only

# Plot the inertia and silhouette score for different cluster counts to determine the optimal number of clusters
# plot_inertia(df, max_clusters=10)
# plot_silhouette(df, max_clusters=10)

total_avg = 0
for _ in range(1):

    # Step 1: K-Means Clustering for General Clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df)
    kmeans_labels = kmeans.labels_
    kmeans_centroids = kmeans.cluster_centers_

    # Prepare for Hierarchical Clustering results and visualization
    # fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    # fig.suptitle("Dendrograms for Hierarchical Clustering within K-Means Clusters", fontsize=16)

    current_label = 0  # To ensure unique labels across all sub-clusters
    hierarchical_labels = np.zeros_like(kmeans_labels, dtype=int)

    
    avg = 0
    for cluster_id in range(5):  # Iterate over K-Means clusters
        cluster_data = df[kmeans_labels == cluster_id]  # Data points in the cluster
        
        if len(cluster_data) > 1:  # Hierarchical clustering requires more than one point
            # Apply Hierarchical Clustering
            linkage_matrix = linkage(cluster_data, method='complete')
            
            # Calculate dynamic distance threshold (e.g., 50% of max distance in the linkage matrix)
            max_distance = np.max(linkage_matrix[:, 2])
            dynamic_t = 0.7 * max_distance  # Adjust fraction as needed
            
            # Use dynamic threshold for fcluster
            sub_cluster_labels = fcluster(linkage_matrix, t=dynamic_t, criterion='distance')
            
            # Plot dendrogram with y-axis labels for distances
            # dendrogram(linkage_matrix, ax=axes[cluster_id], truncate_mode='lastp', p=10, color_threshold=0)
            # axes[cluster_id].set_title(f"Cluster {cluster_id + 1}")
            # axes[cluster_id].set_xlabel("Points")
            # axes[cluster_id].set_ylabel("Distance")
            
            # Ensure y-axis labels are displayed correctly
            # axes[cluster_id].tick_params(axis='y', which='both', left=True, labelleft=True)

            # Calculate Cophenetic Correlation Coefficient (CPCC)
            cophenetic_corr, _ = cophenet(linkage_matrix, pdist(cluster_data))
            avg += cophenetic_corr
            # print(f"Cluster {cluster_id + 1}: CPCC = {cophenetic_corr:.4f}")
        

            # Assign unique labels to hierarchical sub-clusters
            hierarchical_labels[kmeans_labels == cluster_id] = sub_cluster_labels + current_label
            current_label += len(np.unique(sub_cluster_labels))
        # else:
            # axes[cluster_id].set_title(f"Cluster {cluster_id + 1} (Single Point)")
            # axes[cluster_id].axis('off')  # Hide axes for clusters with a single point

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for title
    # plt.show()
    
    avg /= 5
    # print(f"Average CPCC: {avg:.4f}")
    total_avg += avg

total_avg /= 1
print(f"Total Average CPCC: {total_avg:.4f}")

# Visualize the distance matrix and hierarchical cluster assignments
distance_matrix_visualization(df, hierarchical_labels)


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
        distances_to_kmeans_center = np.linalg.norm(subcluster_data - cluster_center, axis=1)
        
        point_outliers.extend(
            subcluster_data[
                (distances_to_subcluster_center > 2 * subcluster_std_dev) &
                (distances_to_kmeans_center > 2 * cluster_std_dev)
            ].index
        )

        # plt.hist(distances_to_subcluster_center, bins=30, alpha=0.7, label='Point-to-Subcluster')
        # plt.hist(distances_to_kmeans_center, bins=30, alpha=0.7, label='Point-to-KMeans')
        # plt.axvline(2 * subcluster_std_dev, color='red', linestyle='--', label='2x Subcluster Std Dev')
        # plt.axvline(2 * cluster_std_dev, color='blue', linestyle='--', label='2x Cluster Std Dev')
        # plt.legend()
        # plt.show()

# Define a set of marker styles for subclusters
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x']

# Create a scatter plot for clusters and subclusters
plt.figure(figsize=(10, 8))

unique_clusters = np.unique(kmeans_labels)
cmap = plt.get_cmap('viridis', len(unique_clusters))  # Create a colormap for K-Means clusters

for cluster_id in unique_clusters:
    cluster_data = df[kmeans_labels == cluster_id]
    subcluster_ids = np.unique(hierarchical_labels[kmeans_labels == cluster_id])

    # Use a consistent color for the cluster, but vary the markers for its subclusters
    cluster_color = cmap(cluster_id / len(unique_clusters))  # Assign color based on the cluster ID
    
    for i, subcluster_id in enumerate(subcluster_ids):
        subcluster_data = cluster_data[hierarchical_labels[kmeans_labels == cluster_id] == subcluster_id]
        marker_style = markers[i % len(markers)]  # Cycle through marker styles
        
        if subcluster_id in outlier_clusters:  # Highlight outlier subclusters
            plt.scatter(
                subcluster_data.iloc[:, 0], subcluster_data.iloc[:, 1],
                edgecolors='red', facecolors=cluster_color,
                marker=marker_style, label=f"Outlier Subcluster {subcluster_id}"
            )
        else:
            plt.scatter(
                subcluster_data.iloc[:, 0], subcluster_data.iloc[:, 1],
                color=cluster_color, marker=marker_style,
                alpha=0.7, label=f"Cluster {cluster_id} - Subcluster {subcluster_id}"
            )

# Highlight point outliers with 'x'
plt.scatter(
    df.iloc[point_outliers, 0], df.iloc[point_outliers, 1],
    c='red', marker='x', label='Point Outliers'
)

# Highlight K-Means centroids
plt.scatter(
    kmeans_centroids[:, 0], kmeans_centroids[:, 1],
    marker='*', s=200, c='black', edgecolors='white', label='K-Means Centroids'
)

plt.title("Scatter Plot with Subcluster Markers, Cluster Colors, and Outlier Highlighting")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# Step 4: Metrics Calculation

# Metrics Calculation
silhouette_avg = silhouette_score(df, hierarchical_labels)
fraction_point_outliers = len(point_outliers) / len(df)
num_total_subclusters = len(np.unique(hierarchical_labels))
fraction_cluster_outliers = len(outlier_clusters) / num_total_subclusters

# Print Metrics
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Fraction of Points Flagged as Outliers: {fraction_point_outliers:.3f}")
print(f"Fraction of Subclusters Flagged as Outliers: {fraction_cluster_outliers:.3f}")