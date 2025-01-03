import pandas as pd
import os
from scipy.io import arff
from sklearn.cluster import KMeans

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
