import pandas as pd
import os
from scipy.io import arff

# code
arff_file = arff.loadarff('PenDigits_withoutdupl_norm_v10.arff')
df = pd.DataFrame(arff_file[0])
print(df.tail())