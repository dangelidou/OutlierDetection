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

kmeans = KMeans(n_clusters=4, random_state=0).fit(df)
print(kmeans.score(df))

# cost =[]
# for i in range(1, 6):
# 	KM = KMeans(n_clusters = i, max_iter = 500)
# 	KM.fit(df)
	
# 	# calculates squared error
# 	# for the clustered points
# 	cost.append(KM.inertia_)	 

# # plot the cost against K values
# plt.plot(range(1, 6), cost, color ='g', linewidth ='3')
# plt.xlabel("Value of K")
# plt.ylabel("Squared Error (Cost)")
# plt.show() # clear the plot

# # the point of the elbow is the 
# # most optimal value for choosing k

pca = PCA(n_components=2, svd_solver='auto')
data_reduced = pca.fit_transform(df)

# Plot the reduced data
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=kmeans.labels_)
plt.show()


