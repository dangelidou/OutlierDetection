This project was developed during the course "Data Warehouses and Data Mining" - CS Department, AUTH
Students: Konstantinos Takis 3518
          Dimitra Angelidou 4200

*********** INSTRUCTIONS & GENERAL INFORMATION *************

- Mall_Customers.csv -> file containing the original dataset
- Mall_Customers_with_outliers.csv -> file containing the final dataset, with seeded outliers

- dataLoading.py -> This file contains: 
                        * Loading of the Data
                        * Outlier addition
                        * Plotting all the features to choose the optimal ones for clustering
                        * Calculation of the silhuette scores to determine the optimal number of clusters for k-means
                        * Visual comparison of the clustering with and without the added outliers
                    In general, all the preprocessing. Doesn't need to run.

- outlierDetection.py -> This file contains the code suggested by ChatGPT (false). Doesn't use the hierarchical algorithm to determine the outliers. 
                         You can run this file if you wish to see how the outlier detection would act based only on the k-means model. 

- outlierDetection2.py -> This file contains the fixed code, uses k-means, then the hierachical algorithm with complete linkage, then detects the outliers and plots them.
                          This is the file you will run to see the final results of our project.

*In the Plots folder you can see most of the project's visualization plots