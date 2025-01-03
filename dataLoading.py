import pandas as pd
import os
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# # code
# arff_file = arff.loadarff('PenDigits_withoutdupl_norm_v10.arff')
# df = pd.DataFrame(arff_file[0])
# print(df.tail())

file_name = 'SpamBase_withoutdupl_norm_10_v10.arff'
df = pd.DataFrame(arff.loadarff(file_name)[0])

# Removed last column becuase it produces error (it has values yes/no)
df = df.iloc[:, :-1]

kmeans = KMeans(n_clusters=2805, random_state=0).fit(df)
print(kmeans.score(df))

pca = PCA(n_components=2, svd_solver='auto')
data_reduced = pca.fit_transform(df)

# Plot the reduced data
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=kmeans.labels_)
plt.show()


