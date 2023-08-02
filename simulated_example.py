"""
Demonstrative Example / Monte Carlo Simulations
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys
import time
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

sns.set_style("whitegrid")
sys.path.append("C:/Users/Marie/OneDrive/Documents/multivariate-causal-forest")


from mcf.multivariate_forest import CausalForest
from sklearn.model_selection import train_test_split


os.chdir("C:/Users/Marie/OneDrive/Documents/multivariate-causal-forest")
np.random.seed(0)

N = 150
t = np.random.binomial(n=1, p=0.5, size=N)

# Simulate 4 covariates from a standard normal distribution
covariate_1 = np.random.normal(loc=0, scale=1, size=N)
covariate_2 = np.random.normal(loc=0, scale=1, size=N)
covariate_3 = np.random.normal(loc=0, scale=1, size=N)
covariate_4 = np.random.normal(loc=0, scale=1, size=N)

X = np.c_[covariate_1, covariate_2, covariate_3, covariate_4]
print("mean treated", t.mean())


epsilon1 = np.array(np.random.normal(0, 1, size = N))
epsilon2 = np.array(np.random.normal(0, 1, size = N))

y1 = 0.5 + t*0.5*X[:, 1] + epsilon1 
y2 = 0.7 + t*X[:, 1]*X[:, 2] + epsilon2


t = np.array(t, dtype = bool)


def train_test(X, t, y1, y2):
    
    ## split into train and estimation samples
    ## stack outcome and X together
    Xy = np.c_[y1, y2, X]
    
    Xy_train, Xy_test, T_train, T_test = train_test_split(
    Xy, np.c_[t], test_size=.21, stratify=np.c_[t], random_state=0)
    
    y1_train = Xy_train[:, 0]
    y1_test = Xy_test[:, 0]
    
        
    y2_train = Xy_train[:, 1]
    y2_test = Xy_test[:, 1]
    
    
    X_train = Xy_train[:, 2:]
    X_test = Xy_test[:, 2:]
    
    t_train = T_train[:, 0]
    t_test = T_test[:, 0]
    
    
    
    # initialize index (the root node considers all observations).
    
    return X_train, X_test, y1_train, y1_test, y2_train, y2_test, t_train, t_test

      
X_train, X_test, y1_train, y1_test, y2_train, y2_test, t_train, t_test = train_test(X, t, y1, y2)

 
       
def evaluate_forests(X_train, X_test, t_train, y1_train, y2_train):
    
    cf = CausalForest(
        num_trees=300,
        split_ratio=0.5,
        min_leaf=5,
        max_depth=20,
        use_transformed_outcomes=False,
        num_workers=3,
        seed_counter=1,    
    )


    cf.fit(X_train, t_train, y1_train, y2_train)

    
    ##predict the model
    ate1, ate2, std_ate1, std_ate2= cf.predict(X_test)
    #treat_effect_train = cf.predict(X_train)

    return ate1, ate2, std_ate1, std_ate2



ate1, ate2, std_y1, std_y2 = evaluate_forests(X_train, X_test, t_train, y1_train, y2_train)

print(ate1)
print(ate2)



############# GROUP AVERAGE POLICY EFFECTS ########################
### Here assume 2 groups (in practice, identify with cross-validation) 
# n_components = 2
# pls_1 = PLSRegression(n_components = n_components)
# pls_1.fit(X_train, y1_train)

# X_scores_train_1 = pls_1.transform(X_train)
# X_scores_test_1 = pls_1.transform(X_test)

# X_scores_train_1 = pls_1.transform(X_train)
# X_scores_test_1 = pls_1.transform(X_test)

# pls_2 = PLSRegression(n_components = n_components)
# pls_2.fit(X_train, y2_train)

# X_scores_train_2 = pls_1.transform(X_train)
# X_scores_test_2 = pls_1.transform(X_test)

# X_scores_train_2 = pls_1.transform(X_train)
# X_scores_test_2 = pls_1.transform(X_test)

# X_scores_train = np.c_[X_scores_train_1, X_scores_train_2]
# X_scores_test = np.c_[X_scores_test_1, X_scores_test_2]


## Run multivariate causal forest
# ate1, ate2, std_y1, std_y2 = evaluate_forests(X_scores_train, X_scores_test, t_train, y1_train, y2_train)

# print(ate1)
# print(ate2)







