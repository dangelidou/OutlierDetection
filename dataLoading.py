import numpy as np
import pandas as pd
import os
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def kmeans_silhoutte_score(data, kmin, kmax):
    silhouette_scores = []
    for k in range(kmin, kmax):  # Test for a range of k values
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    return silhouette_scores

def plot_silhouette_scores(silhouette_scores):
    plt.plot(range(2, 20), silhouette_scores, marker='o')
    plt.title("Silhouette Scores for KMeans Clustering")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

def introduce_outliers(df, percentage_of_outliers):
    # Determine how many outliers to create
    n_outliers = int(len(df) * percentage_of_outliers)  # 5% of 200 = 10

    # Interval at which to insert new rows
    interval = len(df) // n_outliers  # 200 // 10 = 20

    # Prepare a list to store new (outlier) rows
    outlier_rows = []

    # Define possible outlier ranges (customizable)
    age_outliers = [lambda: np.random.randint(1, 10),   # extremely low age
                    lambda: np.random.randint(80, 101)] # extremely high age
    income_outliers = [lambda: np.random.randint(150, 221), # very high income
                    lambda: np.random.randint(0, 5)]     # near 0 income
    score_outliers = [lambda: np.random.randint(110, 161),  # above 100
                    lambda: np.random.randint(0, 5)]      # near 0

    # Loop to create outliers
    for i in range(n_outliers):
        # Pick the row around which we want to create an outlier (e.g., row i*interval)
        base_index = i * interval
        row = df.iloc[base_index].copy()

        # Assign a new CustomerID
        row["CustomerID"] = df["CustomerID"].max() + i + 1

        # Randomly pick whether each field is going to be a low or high outlier
        row["Age"] = np.random.choice(age_outliers)()
        row["Annual Income (k$)"] = np.random.choice(income_outliers)()
        row["Spending Score (1-100)"] = np.random.choice(score_outliers)()

        # (Optional) flip gender randomly for the new row
        row["Gender"] = np.random.choice(["Male", "Female"])

        outlier_rows.append(row)

    # Combine original dataset with outlier rows
    df_outliers = pd.concat([df, pd.DataFrame(outlier_rows)], ignore_index=True)

    # Sort by CustomerID just for neatness (optional)
    df_outliers.sort_values(by="CustomerID", inplace=True)
    

    return df_outliers

def with_without_outlier_scatterplot(df, df_outlier):
    # Define number of clusters
    k = 5

    # Fit k-means on data WITHOUT outlier
    kmeans_no_outlier = KMeans(n_clusters=k, random_state=0)
    kmeans_no_outlier.fit(df)
    mse_no_outlier = kmeans_no_outlier.inertia_ / len(df)

    # Fit k-means on data WITH outlier
    kmeans_with_outlier = KMeans(n_clusters=k, random_state=0)
    kmeans_with_outlier.fit(df_outliers)
    mse_with_outlier = kmeans_with_outlier.inertia_ / len(df_outliers)

    # Create a figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # -----------------
    # Subplot 1: Without outlier
    axes[0].scatter(df.iloc[:, 0], df.iloc[:, 1], c=kmeans_no_outlier.labels_, cmap='viridis', alpha=0.7)
    axes[0].scatter(
        kmeans_no_outlier.cluster_centers_[:, 0],
        kmeans_no_outlier.cluster_centers_[:, 1],
        marker='*', s=200, c='red', edgecolors='black', linewidths=1.5
    )
    axes[0].set_title(f'Without Outlier\nMSE = {mse_no_outlier:.2f}')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # -----------------
    # Subplot 2: With outlier
    axes[1].scatter(df_outliers.iloc[:, 0], df_outliers.iloc[:, 1], c=kmeans_with_outlier.labels_, cmap='viridis', alpha=0.7)
    axes[1].scatter(
        kmeans_with_outlier.cluster_centers_[:, 0],
        kmeans_with_outlier.cluster_centers_[:, 1],
        marker='*', s=200, c='red', edgecolors='black', linewidths=1.5
    )
    axes[1].set_title(f'With Outlier\nMSE = {mse_with_outlier:.2f}')
    axes[1].set_xlabel('Feature 1')

    plt.tight_layout()
    plt.show()


# Load the dataset
file_name = 'Mall_Customers.csv'
df = pd.read_csv(file_name)

# Introduce outliers
df_outliers = introduce_outliers(df, percentage_of_outliers=0.05)

# kept the last three columns which are the ones that define the clusters
df = df.iloc[:, 3:]
df_outliers = df_outliers.iloc[:, 3:]

# pca = PCA(n_components=2, svd_solver='auto')
# data_reduced = pca.fit_transform(df_scaled)

# Calculating silhouette score to find the optimal number of clusters
silhouette_scores = kmeans_silhoutte_score(df, 2, 20)
plot_silhouette_scores(silhouette_scores)


# plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=kmeans.labels_)
# plt.title("5-Means Clustering of Mall Customers")
# plt.xlabel("Annual Income (K$)")
# plt.ylabel("Spending Score (1-100)")
# plt.show()