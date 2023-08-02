"""
Module to fit a multivariate causal tree.

This module provides functions to fit a causal tree and predict treatment
effects using a fitted causal tree.
"""
from itertools import count
import numpy as np
import pandas as pd
from numba import njit
import os 
from sklearn.model_selection import train_test_split


def _honest_splitting(X, t, y1, y2):
    
    """
    Args: 
    
        X (np.array): Data on features
        t (np.array): Data on treatment status
        y1 (np.array): Data on outcome 1
        y2 (np.array): Data on outcome 2
    

    Returns:
        X_train  (np.array): Train features used to build the tree 
        X_est    (np.array): Estimation data used to estimate the policy effect
        y1_train (np.array): Train outcome 1 used to build the tree 
        y1_est   (np.array): Estimation outcome 1 used to estimate the policy effect
        y2_train (np.array): Train outcome 2 used to build the tree 
        y2_est   (np.array): Estimation outcome 2 used to estimate the policy effect
        t_train  (np.array): Train treatment (policy) used to build the tree 
        t_est    (np.array): Estimation treatment (policy) used to estimate the policy effect 
  
    """
    
    ## split into train and estimation samples
    ## stack outcome and X together
    Xy = np.c_[y1, y2,  X]
    
    Xy_train, Xy_est, t_train, t_est = train_test_split(
    Xy, t, test_size=.50, stratify=t, random_state=42)
    
    y1_train = Xy_train[:, 0]
    y1_est = Xy_est[:, 0]
    
    y2_train = Xy_train[:, 1]
    y2_est = Xy_est[:, 1]
    
    X_train = Xy_train[:, 2:]
    X_est = Xy_est[:, 2:]
    
    # initialize index (the root node considers all observations).
    n = len(y1_train)
    index_est = np.full((n,), True)
    
    return X_train, X_est, y1_train, y1_est, y2_train, y2_est, t_train, t_est, index_est





def fit_causaltree(X, t, y1, y2, critparams=None, honesty = True):
    """Fit a causal tree on given data.

    Wrapper function for `_fit_node`. Sets default parameters for
    *crit_params* and calls internal fitting function `_fit_node`. Returns
    fitted tree as a pd.DataFrame.

    Args:
        X (np.array): Data on features
        t (np.array): Data on treatment status
        y1 (np.array): Data on outcomes
        y2 (np.array): Data on outcomes
        critparams (dict): Dictionary containing information on when to stop
            splitting further, i.e., minimum number of leafs and maximum
            depth of the tree. Default is set to 'min_leaf' = 4 and
            'max_depth' = 20.
        honesty (bool): Boolean value. If True, honest estimation is done

    Returns:
        ctree (pd.DataFrame): the fitted causal tree represented as a pandas
            data frame. If honest is true ctree is an honest causal tree and
            a regular otherwise.

    """
    if critparams is None:
        critparams = {
            "min_leaf": 4,
            "max_depth": 20,
            "use_transformed_outcomes": False,
        }
        

    # initialize counter object and id_params
    counter = count(0)
    rootid = next(counter)
    idparams = {"counter": counter, "id": rootid, "level": 0}

    # initialize index (the root node considers all observations).
    n = len(y1)
    index = np.full((n,), True)
    

    if honesty == False:
        # fit tree
        ctree_array = _fit_node(
            X=X, t=t, y1=y1, y2 = y2, index=index, critparams=critparams, idparams=idparams, honesty = False, t_est = None, 
            y_est = None,
        )
    else:
        
        hnst = _honest_splitting(X, t, y1, y2)
        X_train, X_est, y1_train, y1_est, y2_train, y2_est, t_train, t_est, index_est = hnst
        
        ctree_array = _fit_node(
                    X=X_train, t=t_train, y1=y1_train, y2 = y2_train, index=index_est, critparams=critparams, 
                    idparams=idparams, honesty = True, t_est = t_est, y1_est = y1_est, y2_est = y2_est,
        )

    column_names = [
        "id",
        "left_child",
        "right_child",
        "level",
        "split_feat",
        "split_value",
        "treat_effect_1",
        "treat_effect_2",
    ]
    columns_to_int = column_names[:5]

    ct = pd.DataFrame(ctree_array, columns=column_names)
    ct[columns_to_int] = ct[columns_to_int].astype("Int64")
    ct = ct.set_index("id").sort_index()
    return ct


def _fit_node(X, t, y1, y2, index, critparams, idparams, honesty, t_est, y1_est, y2_est):
    """Fits a single decision tree node recursively.

    Recursively split feature space until stopping criteria in *crit_params*
    are reached. In each level of recursion fit a single node.

    Args:
        X (np.array): data on features
        t (np.array): data on treatment status
        y1 (np.array): data on outcome 1
        y2 (np.array): data on outcome 2
        index (np.array): boolean index indicating which observations (rows)
            of the data to consider for split.
        critparams (dict): dictionary containing information on when to stop
            splitting further, i.e., minimum number of leafs and maximum
            depth of the tree, and if the transformed outcome should be used.
        idparams (dict): dictionary containing identification information of
            a single node. That is, a unique id number, the level in the tree,
            and a counter object which is passed to potential children of node.

    Returns:
        out (np.array): array containing information on the splits, with
            columns representing (nodeid, left_childid, right_childid, level,
            split_feat).

    """
    level = idparams["level"]
    nodeid = idparams["id"]
            
    tmp = _find_optimal_split(
        X=X,
        t=t,
        y1=y1,
        y2=y2,
        index=index,
        min_leaf=critparams["min_leaf"],
        use_transformed_outcomes=critparams["use_transformed_outcomes"],
        honesty = honesty,
        t_est = t_est,
    )

    if tmp is None or level == critparams["max_depth"]:
        # if we do not split the node must be a leaf, hence we add the
        # treatment effect to the output.
        if honesty == False:
            treat_effect_1, treat_effect_2 = _estimate_treatment_effect(y1[index], y2[index], t[index])
        else:
            treat_effect_1, treat_effect_2 = _estimate_treatment_effect(y1_est[index], y2_est[index], t_est[index])

        info = np.array(
            [nodeid, np.nan, np.nan, level, np.nan, np.nan, treat_effect_1, treat_effect_2, ]
        ).reshape((1, 8))
        return info
    else:
        left, right, split_feat, split_value = tmp

        leftid = next(idparams["counter"])
        rightid = next(idparams["counter"])

        info = np.array([nodeid, leftid, rightid, level])

        split_info = np.array([split_feat, split_value, np.nan, np.nan])
        info = np.append(info, split_info).reshape((1, 8))

        idparams_left = idparams.copy()
        idparams_left["id"] = leftid
        idparams_left["level"] += 1

        idparams_right = idparams.copy()
        idparams_right["id"] = rightid
        idparams_right["level"] += 1

        out_left = _fit_node(
            X=X,
            t=t,
            y1=y1,
            y2=y2,
            index=left,
            critparams=critparams,
            idparams=idparams_left,
            honesty = honesty,
            y1_est = y1_est,
            y2_est = y2_est,
            t_est = t_est,
        )
        out_right = _fit_node(
            X=X,
            t=t,
            y1=y1,
            y2=y2,
            index=right,
            critparams=critparams,
            idparams=idparams_right,
            honesty = honesty,
            y1_est = y1_est,
            y2_est = y2_est,
            t_est = t_est,

        )

        out = np.vstack((info, out_left))
        out = np.vstack((out, out_right))
        return out



def _find_optimal_split(X, t, y1, y2,  index, min_leaf, use_transformed_outcomes, honesty, t_est):
    """Compute optimal split and splitting information.

    For given data, for each feature, go through all valid splitting points and
    find the split value and split variable which result in the lowest overall
    loss.

    Args:
        X (np.array): data on features
        t (np.array): data on treatment status
        y1 (np.array): data on outcome 1
        y2 (np.array): data on outcome 2
        index (np.array): boolean index indicating which observations (rows)
            of the data to consider for split.
        min_leaf (int): Minimum number of observations of each type (treated,
            untreated) allowed in a leaf; has to be greater than 1.
        use_transformed_outcomes (bool): Use transformed outcomes for loss
            approximation or simply return sum of squared estimates.

    Returns:
        left (np.array): boolean index representing the observations falling
            in the left leaf.
        right (np.array): boolean index representing the observations falling
            in the right leaf.
        split_feat (int): index of feature on which the optimal split occurs.
        split_value (float): value of feature on which the optimal split
            occurs.

    """
    _, p = X.shape
    split_feat = None
    split_value = None
    split_index = None
    loss = np.inf if use_transformed_outcomes else -np.inf
           
    
    for j in range(p):
        # loop through features

        index_sorted = np.argsort(X[index, j])
        yy1 = y1[index][index_sorted]
        yy2 = y2[index][index_sorted]
        xx = X[index, j][index_sorted]
        tt = t[index][index_sorted]
        
        if honesty == True:  
            tt_est = t_est[index][index_sorted]
        else: 
            tt_est = None
        

        yy1_transformed, yy2_transformed = _transform_outcome(y1=yy1, y2 = yy2, t=tt)

        splitting_indices = _compute_valid_splitting_indices(
            t=tt, min_leaf=min_leaf, honesty=honesty, t_est = tt_est

        )

        # loop through observations
        tmp = _find_optimal_split_inner_loop(
            splitting_indices=splitting_indices,
            x=xx,
            t=tt,
            y1=yy1,
            y2=yy2,
            y1_transformed=yy1_transformed,
            y2_transformed=yy2_transformed,
            min_leaf=min_leaf,
            use_transformed_outcomes=use_transformed_outcomes,
            honesty = honesty,
        )
        jloss, jsplit_value, jsplit_index = tmp

        if use_transformed_outcomes:
            if jloss < loss and jsplit_index is not None:
                split_feat = j
                split_value = jsplit_value
                split_index = jsplit_index
                loss = jloss
        else:
            if jloss > loss and jsplit_index is not None:
                split_feat = j
                split_value = jsplit_value
                split_index = jsplit_index
                loss = jloss

    # check if any split has occured.
    if loss == np.inf or loss == -np.inf:
        return None

    # create index of observations falling in left and right leaf, respectively
    index_sorted = np.argsort(X[index, split_feat])
    
    left, right = _retrieve_index(
            index=index, sorted_subset_index=index_sorted, split_index=split_index
            )

    return left, right, split_feat, split_value





@njit
def _find_optimal_split_inner_loop(
    splitting_indices,
    x,
    t,
    y1,
    y2,
    y1_transformed,
    y2_transformed,
    min_leaf,
    use_transformed_outcomes,
    honesty = True
):
    """Find the optimal splitting value for data on a single feature.

    Finds the optimal splitting value (in the data) given data on a single
    feature. Note that the algorithm is essentially not different from naively
    searching for a minimum; however, since this is computationally very costly
    this function implements a dynamic updating procedure, in which the sums
    get updated during the loop. See "Elements of Statistical Learning" for a
    reference on the classical algorithm.

    Args:
        splitting_indices (np.array): valid splitting indices.
        x (np.array): data on a single feature.
        t (np.array): data on treatment status.
        y1 and y2 (np.array): data on outcomes.
        y1_transformed and y2_transformed (np.array): data on transformed outcomes.
        use_transformed_outcomes (bool): Use transformed outcomes for loss
            approximation or simply return sum of squared estimates.

    Returns:
         - (np.inf, None, None): if *splitting_indices* is empty
         - (minimal_loss, split_value, split_index): if *splitting_indices* is
            not empty, where minimal_loss denotes the loss occured when
            splitting the feature axis at the split_value (= x[split_index]).

    """
    
    n_obs = len(y1)

    if len(splitting_indices) == 0:
        return np.inf, None, None

    # initialize number of observations
    i0 = splitting_indices[0]
    n_1l = int(np.sum(t[: (i0 + 1)]))
    n_0l = int(np.sum(~t[: (i0 + 1)]))
    n_1r = int(np.sum(t[(i0 + 1) :]))
    n_0r = len(t) - n_1l - n_0l - n_1r

    # initialize dynamic sums
    sum1_1l = y1[t][: (i0 + 1)].sum()
    sum1_0l = y1[~t][: (i0 + 1)].sum()
    sum1_1r = y1[t][(i0 + 1) :].sum()
    sum1_0r = y1[~t][(i0 + 1) :].sum()


    sum2_1l = y2[t][: (i0 + 1)].sum()
    sum2_0l = y2[~t][: (i0 + 1)].sum()
    sum2_1r = y2[t][(i0 + 1) :].sum()
    sum2_0r = y2[~t][(i0 + 1) :].sum()

    
    # compute means
    if n_1l > 0:
        mu1_1l = sum1_1l/n_1l 
        mu2_1l = sum2_1l/n_1l 
    else:
        mu1_1l = 0
        mu2_1l = 0
        
        
    if n_0l > 0:
        mu1_0l = sum1_0l/n_0l 
        mu2_0l = sum2_0l/n_0l 
    else:
        mu1_0l = 0
        mu2_0l = 0

    if n_1r >= 1:
        mu1_1r = sum1_1r/n_1r 
        mu2_1r = sum2_1r/n_1r     
    else:
        mu1_1r = 0
        mu2_1r = 0
    
    if n_0r > 0:
        
        mu1_0r = sum1_0r/n_0r
        mu2_0r = sum2_0r/n_0r 

    else: 
        
        mu1_0r = 0
        mu2_0r = 0
    
    
    # # square and then sum outcomes
    
    sumsq1_1l = np.sum((y1[t][: (i0 + 1)]- mu1_1l)**2)
    sumsq1_0l = np.sum((y1[~t][: (i0 + 1)]- mu1_0l)**2)
    sumsq1_1r = np.sum((y1[t][(i0 + 1) :] - mu1_1r)**2)
    sumsq1_0r = np.sum((y1[~t][(i0 + 1) :] - mu1_0r)**2)
    
    
    
    sumsq2_1l = np.sum((y2[t][: (i0 + 1)]- mu2_1l)**2)
    sumsq2_0l = np.sum((y2[~t][: (i0 + 1)]- mu2_0l)**2)
    sumsq2_1r = np.sum((y2[t][(i0 + 1) :] - mu2_1r)**2)
    sumsq2_0r = np.sum((y2[~t][(i0 + 1) :] - mu2_0r)**2)
    
    
    ##initialize sum for covariance
    sumy1y2_1l = (y1[t][: (i0 + 1)]*y2[t][: (i0 + 1)]).sum()
    sumy1y2_0l = (y1[~t][: (i0 + 1)]*y2[~t][: (i0 + 1)]).sum()
    sumy1y2_1r = (y1[t][(i0 + 1) :]*y2[t][(i0 + 1) :]).sum()
    sumy1y2_0r = (y1[~t][(i0 + 1) :]*y2[~t][(i0 + 1) :]).sum()
    
    
    
    ## compute fraction of treated
        
    
    split_value = x[i0]
    split_index = i0
    minimal_loss = _compute_global_loss(
        sum1_0l=sum1_0l, sum1_1l=sum1_1l, sum1_0r=sum1_0r, sum1_1r=sum1_1r,
        sum2_0l=sum2_0l, sum2_1l=sum2_1l, sum2_0r=sum2_0r, sum2_1r=sum2_1r,
        
        n_0l=n_0l, n_1l=n_1l, n_0r=n_0r, n_1r=n_1r,
        
        
        sumsq1_1l = sumsq1_1l, sumsq1_0l = sumsq1_0l, sumsq1_1r = sumsq1_1r, sumsq1_0r = sumsq1_0r, 
        sumsq2_1l = sumsq2_1l, sumsq2_0l = sumsq2_0l, sumsq2_1r = sumsq2_1r, sumsq2_0r = sumsq2_0r,
        
        
        sumy1y2_1l = sumy1y2_1l, sumy1y2_0l = sumy1y2_0l, sumy1y2_1r= sumy1y2_1r, sumy1y2_0r = sumy1y2_0r,
        n_obs = n_obs,
        )
    

    for i in splitting_indices[1:]:
        if t[i]:
            sum1_1l += y1[i]
            sum1_1r -= y1[i]
            sum2_1l += y2[i]
            sum2_1r -= y2[i]
            n_1l += 1
            n_1r -= 1
            
            
            # update means
            
            if n_1l > 0:
                mu1_1l = sum1_1l/n_1l 
                mu2_1l = sum2_1l/n_1l 
            else:
                mu1_1l = 0
                mu2_1l = 0
                
                
           
            if n_1r >= 1:
                mu1_1r = sum1_1r/n_1r 
                mu2_1r = sum2_1r/n_1r     
            else:
                mu1_1r = 0
                mu2_1r = 0
            
            
    
    
            
            # # update squares
    
            sumsq1_1l = np.sum((y1[t][: (i + 1)]- mu1_1l)**2)
            sumsq1_1r = np.sum((y1[t][(i + 1) :] - mu1_1r)**2)
            sumsq2_1l = np.sum((y2[t][: (i + 1)]- mu2_1l)**2)
            sumsq2_1r = np.sum((y2[t][(i + 1) :] - mu2_1r)**2)
    


            ## update for covariance
            sumy1y2_1l = (y1[t][: (i + 1)]*y2[t][: (i + 1)]).sum()
            sumy1y2_1r = (y1[t][(i + 1) :]*y2[t][(i + 1) :]).sum()
            
    
        else:
            sum1_0l += y1[i]
            sum1_0r -= y1[i]
            sum2_0l += y2[i]
            sum2_0r -= y2[i]
            n_0l += 1
            n_0r -= 1
            
            if n_0l > 0:
                 mu1_0l = sum1_0l/n_0l 
                 mu2_0l = sum2_0l/n_0l 
            else:
                 mu1_0l = 0
                 mu2_0l = 0
                 
            if n_0r > 0:
                
                mu1_0r = sum1_0r/n_0r
                mu2_0r = sum2_0r/n_0r 

            else: 
                
                mu1_0r = 0
                mu2_0r = 0
            
            # ## update squares
            sumsq1_0l = np.sum((y1[~t][: (i + 1)]- mu1_0l)**2)
            sumsq1_0r = np.sum((y1[~t][(i + 1) :] - mu1_0r)**2)
            sumsq2_0l = np.sum((y2[~t][: (i + 1)]- mu2_0l)**2)
            sumsq2_0r = np.sum((y2[~t][(i + 1) :] - mu2_0r)**2)



            ## update for cov
            sumy1y2_0l = (y1[~t][: (i + 1)]*y2[~t][: (i + 1)]).sum()
            sumy1y2_0r = (y1[~t][(i + 1) :]*y2[~t][(i + 1) :]).sum()


    


        # this should not happen but it does for some reason (Bug alarm!)
        if n_0r < min_leaf or n_1r < min_leaf:
            break
        if n_0l < min_leaf or n_1l < min_leaf:
            continue

        global_loss =_compute_global_loss(
            sum1_0l=sum1_0l, sum1_1l=sum1_1l, sum1_0r=sum1_0r, sum1_1r=sum1_1r,
            sum2_0l=sum2_0l, sum2_1l=sum2_1l, sum2_0r=sum2_0r, sum2_1r=sum2_1r,
            
            n_0l=n_0l, n_1l=n_1l, n_0r=n_0r, n_1r=n_1r,
            
            sumsq1_1l = sumsq1_1l, sumsq1_0l = sumsq1_0l, sumsq1_1r = sumsq1_1r, sumsq1_0r = sumsq1_0r, 
            sumsq2_1l = sumsq2_1l, sumsq2_0l = sumsq2_0l, sumsq2_1r = sumsq2_1r, sumsq2_0r = sumsq2_0r,

            sumy1y2_1l = sumy1y2_1l, sumy1y2_0l = sumy1y2_0l, sumy1y2_1r= sumy1y2_1r, sumy1y2_0r = sumy1y2_0r,

            n_obs = n_obs,
            )
        
        
        if use_transformed_outcomes:
            if global_loss < minimal_loss:
                split_value = x[i]
                split_index = i
                minimal_loss = global_loss
        else:
            if global_loss > minimal_loss:
                split_value = x[i]
                split_index = i
                minimal_loss = global_loss

    return minimal_loss, split_value, split_index


@njit
def _compute_global_loss(
     sum1_1l, sum1_0l, sum1_1r, sum1_0r,
     sum2_1l, sum2_0l, sum2_1r, sum2_0r,
     n_1l, n_0l, n_1r, n_0r,
     sumsq1_1l, sumsq1_0l, sumsq1_1r, sumsq1_0r, 
     sumsq2_1l, sumsq2_0l, sumsq2_1r, sumsq2_0r, 


     sumy1y2_1l, sumy1y2_0l, sumy1y2_1r, sumy1y2_0r,
     
     n_obs,
):
    """Compute global loss when splitting at index *i*.

    Computes global loss when splitting the observation set at index *i*
    using the dynamically updated sums and number of observations.

    Args:
        sum1_1l and sum2_1l (float): Sum of outcomes of treated observations left to the
            potential split at index *i*.
        n_1l (int): Number of treated observations left to the potential split
            at index *i*.
        sum1_0l and sum2_0l (float): Sum of outcomes of untreated observations left to the
            potential split at index *i*.
        n_0l (int): Number of untreated observations left to the potential
            split at index *i*.
        sum1_1r and sum2_1r (float): Sum of outcomes of treated observations right to the
            potential split at index *i*.
        n_1r (int): Number of treated observations right to the potential split
            at index *i*.
        sum1_0r and sum2_0r (float): Sum of outcomes of untreated observations right to the
            potential split at index *i*.
        n_0r (int): Number of untreated observations right to the potential
            split at index *i*.
        y1_transformed and y2_transformed (np.array): Transformed outcomes.
        i (int): Index at which to split.
        use_transformed_outcomes (bool): Use transformed outcomes for loss
            approximation or simply return sum of squared estimates.

    Returns:
        global_loss (float): The loss when splitting at index *i*.

    """
    ate1_l, ate2_l = _compute_treatment_effect_raw(sum1_1l, n_1l, sum1_0l, n_0l, sum2_1l, sum2_0l)
    ate1_r, ate2_r = _compute_treatment_effect_raw(sum1_1r, n_1r, sum1_0r, n_0r, sum2_1r, sum2_0r)
            
    
    left_te = np.array([ate1_l,  ate2_l]).transpose()
    right_te = np.array([ate1_r, ate2_r]).transpose()

        
        
    ## variance
    
    if (n_1l)*((n_1l-1))*(n_1l) > 1 :

        var1_1l = (1/n_1l)*(1/(n_1l-1))*(sumsq1_1l)
        var2_1l = (1/n_1l)*(1/(n_1l-1))*(sumsq2_1l)
        cov_1l = (1/n_1l)*(1/(n_1l-1))*(sumy1y2_1l - (1/n_1l)*sum1_1l*sum2_1l)
    
    else:
        
        var1_1l = 0.01
        var2_1l = 0.01
        cov_1l = 0.0
         
        
    if (n_1r)*((n_1r-1))*(n_1r) > 1 :
        
        var1_1r = (1/n_1r)*(1/(n_1r-1))*(sumsq1_1r)
        var2_1r = (1/n_1r)*(1/(n_1r-1))*(sumsq2_1r)
        cov_1r = (1/n_1r)*(1/(n_1r-1))*(sumy1y2_1r - (1/n_1r)*sum1_1r*sum2_1r) 


    else:
        var1_1r = 0.01
        var2_1r = 0.01
        cov_1r = 0


        


    if (n_0l)*((n_0l-1))*(n_0l) > 1:

    
        var1_0l = (1/n_0l)*(1/(n_0l-1))*(sumsq1_0l)
        var2_0l = (1/n_0l)*(1/(n_0l-1))*(sumsq2_0l)
        cov_0l = (1/n_0l)*(1/(n_0l-1))*(sumy1y2_0l - (1/n_0l)*sum1_0l*sum2_0l)
        
        
    else: 
        
        var1_0l = 0.01
        var2_0l = 0.01
        cov_0l = 0
        
    if (n_0r)*((n_0r-1))*(n_0r) > 1:
        
        var1_0r = (1/n_0r)*(1/(n_0r-1))*(sumsq1_0r)
        var2_0r = (1/n_0r)*(1/(n_0r-1))*(sumsq2_0r)
        cov_0r = (1/n_0r)*(1/(n_0r-1))*(sumy1y2_0r - (1/n_0r)*sum1_0r*sum2_0r)


    else:
        
        var1_0r = 0.01
        var2_0r = 0.01
        cov_0r = 0


    
    var1_l = var1_1l + var1_0l 
    var1_r = var1_1r + var1_0r
    
    var2_l = var2_1l + var2_0l 
    var2_r = var2_1r + var2_0r
    
   
    cov_l = cov_1l + cov_0l
    cov_r = cov_1r + cov_0r

    

    Vcov_l = np.array([[var1_l, cov_l],
                       [cov_l, var2_l]])
     
    Vcov_r = np.array([[var1_r, cov_r],
                       [cov_r, var2_r]])
     
    n_l = n_0l + n_1l
    n_r = n_0r + n_1r
    
    
   
    left_loss =  (n_l/n_obs) * np.dot(np.dot(left_te.transpose(), np.linalg.pinv(Vcov_l)), 
                                                     left_te)  
    right_loss = (n_r/n_obs) * np.dot(np.dot(right_te.transpose(), np.linalg.pinv(Vcov_r)), 
                                                     right_te) 
       

    global_loss = left_loss + right_loss
    return global_loss

@njit
def _compute_valid_splitting_indices(t,  min_leaf, t_est, honesty):
    """Compute valid split indices for treatment array *t* given *min_leaf*.

    Given an array *t* of treatment status and an integer *min_leaf* --denoting
    the minimum number of allowed observations of each type in a leaf node--
    computes a sequence of indices on which we can split *t* and get that each
    resulting side contains a minimum of *min_leaf* treated and untreated
    observations. Returns an empty sequence if no split is possible.

    Args:
        t (np.array): 1d array containing the treatment status as treated =
            True and untreated = False.
        t_est (np.array): 1d array containing the treatment status as treated =
            True and untreated = False (use this when honesty is true).
        min_leaf (int): Minimum number of observations of each type (treated,
            untreated) allowed in a leaf; has to be greater than 1.
        honesty (if honest splitting, then True, else False).

    Returns:
        out (np.array): a sequence of indices representing valid splitting
            points.

    """
    out = np.arange(0)
    
    if honesty == False: 
  
        n = len(t)
        if n < 2 * min_leaf:
            return out

        # find first index at which *min_leaf* treated obs. are in left split
        left_index_treated = np.argmax(np.cumsum(t) == min_leaf)
        if left_index_treated == 0:
            return out
        

        # find first index at which *min_leaf* untreated obs. are in left split
        left_index_untreated = np.argmax(np.cumsum(~t) == min_leaf)
        if left_index_untreated == 0:
            return out
        
        
        # first split at which both treated and untreated occure more often than
        # *min_leaf* is given by the maximum.
        tmparray = np.array([left_index_treated, left_index_untreated])
        left = np.max(tmparray)
        
        
        # do the same for right side
        right_index_treated = np.argmax(np.cumsum(t[::-1]) == min_leaf)
        if right_index_treated == 0:
            return out

        right_index_untreated = np.argmax(np.cumsum(~t[::-1]) == min_leaf)
        if right_index_untreated == 0:
            return out

        tmparray = np.array([right_index_treated, right_index_untreated])
        right = n - np.max(tmparray)
        
        
        if left > right - 1:
                return out
        else:
            out = np.arange(left, right - 1)
            return out
    
    else: 
        
        
        n = len(t_est)
        if n < 2 * min_leaf:
            return out
                

        # find first index at which *min_leaf* treated obs. are in left split
        left_index_treated = np.argmax(np.cumsum(t) == min_leaf)
        if left_index_treated == 0:
            return out
        
        # compute first index where min_leaf treated obs. in the left split using the estimation sample
        left_index_treated_est = np.argmax(np.cumsum(t_est) == min_leaf)
        if left_index_treated_est == 0:
            return out
        

        # find first index at which *min_leaf* untreated obs. are in left split
        left_index_untreated = np.argmax(np.cumsum(~t) == min_leaf)
        if left_index_untreated == 0:
            return out
        
        # find first index at which *min_leaf* untreated obs. are in left split in the estimation sample
        left_index_untreated_est = np.argmax(np.cumsum(~t_est) == min_leaf)
        if left_index_untreated_est == 0:
            return out
        
        
        # first split at which both treated and untreated occure more often than
        # *min_leaf* is given by the maximum of all values.
        tmparray = np.array([left_index_treated,left_index_treated_est, 
                             left_index_untreated, left_index_untreated_est])
        left = np.max(tmparray)
        
        
        # do the same for right side
        right_index_treated = np.argmax(np.cumsum(t[::-1]) == min_leaf)
        if right_index_treated == 0:
            return out
        
        ## compute first index where min_leaf is present on the right using the estimation sample
        right_index_treated_est = np.argmax(np.cumsum(t_est[::-1]) == min_leaf)
        if right_index_treated_est == 0:
            return out
        

        right_index_untreated = np.argmax(np.cumsum(~t[::-1]) == min_leaf)
        if right_index_untreated == 0:
            return out
        
        
        right_index_untreated_est = np.argmax(np.cumsum(~t_est[::-1]) == min_leaf)
        if right_index_untreated_est == 0:
            return out


        tmparray = np.array([right_index_treated,right_index_treated_est,
                             right_index_untreated, right_index_untreated_est])
        
        right = n - np.max(tmparray) 
        
        
        if left > right - 1:
                return out
        else:
            out = np.arange(left, right - 1)
            return out
     

@njit
def _transform_outcome(y1, y2, t):
    """Transform outcome.

    Transforms outcome using approximate propensity scores. Equation is as
    follows: y_transformed_i = 2 * y_i * t_i - 2 * y_i * (1 - t_i), where t_i
    denotes the treatment status of the ith individual. This object is
    equivalent to the individual treatment effect in expectation.

    Args:
        y1, y2 (np.array): data on outcomes.
        t (np.arra): boolean data on treatment status.

    Returns:
        y1_transformed, y2_transformed (np.array): the transformed outcome.

    Example:
    >>> import numpy as np
    >>> y = np.array([-1, 0, 1])
    >>> t = np.array([True, True, False])
    >>> _transform_outcome(y, t)
    array([-2,  0, -2])

    """
    y1_transformed = 2 * y1 * t - 2 * y1 * (1 - t)
    y2_transformed = 2 * y2 * t - 2 * y2 * (1 - t)

    return y1_transformed, y2_transformed


@njit
def _estimate_treatment_effect(y1, y2,  t):
    """Estimate the average treatment effect.

    Estimates average treatment effect (ATE) using outcomes *y* and treatment
    status *t*.

    Args:
        y1, y2 (np.array): data on outcomes.
        t (np.array): boolean data on treatment status.

    Returns:
        out (float): the estimated treatment effect.

    Example:
    >>> import numpy as np
    >>> y = np.array([-1, 0, 1, 2, 3, 4])
    >>> t = np.array([False, False, False, True, True, True])
    >>> _estimate_treatment_effect(y, t)
    3.0

    """
    ate1 = y1[t].mean() - y1[~t].mean()
    ate2 = y2[t].mean() - y2[~t].mean()

    return ate1, ate2


def _retrieve_index(index, sorted_subset_index, split_index):
    """Get index of left and right leaf relative to complete data set.

    Given an array of indices *index* of length of the original data set, and
    a sorted index array *index_sorted* (sorted with respect to the feature
    on which we split; see function _find_optimal_split) and an index on which
    we want to split (*split_index*), `_retrieve_index` computes two indices
    (left and right) the same length as *index* corresponding to observations
    falling falling left and right to the splitting point, respectively.

    Args:
        index (np.array): boolean array indicating which observations (rows)
            of the data to consider for split.
        sorted_subset_index (np.array): array containing indices, sorted with
            respect to the feature under consideration. Length is equal to the
            number of True values in *index*.
        split_index (int): index in *sorted_subset_index* corresponding to the
            split.

    Returns:
        out: 2d tuple containing np.arrays left_index and right_index the same
            length as *index*

    Example:
    >>> import numpy as np
    >>> from pprint import PrettyPrinter
    >>> index = np.array([True, True, True, False, False, True])
    >>> sorted_subset_index = np.array([0, 3, 1, 2])
    >>> split_index = 1
    >>> PrettyPrinter().pprint(_retrieve_index(index, sorted_subset_index,
    ... split_index))
    (array([ True, False, False, False, False,  True]),
     array([False,  True,  True, False, False, False]))

    """
    # Not solving the bug:
    if split_index is None:
        return index

    left = sorted_subset_index[: (split_index + 1)]
    right = sorted_subset_index[(split_index + 1) :]
    nonzero_index = np.nonzero(index)[0]

    # initialize new indices
    n = len(index)
    left_index = np.full((n,), False)
    right_index = np.full((n,), False)

    # fill nonzero values
    left_index[nonzero_index[left]] = True
    right_index[nonzero_index[right]] = True

    out = left_index, right_index
    return out


@njit
def _compute_treatment_effect_raw(
    sum_treated, n_treated, sum_untreated, n_untreated, sum2_treated, sum2_untreated
):
    """Compute the average treatment effect.

    Computes the average treatment effect (ATE) using the sum of outcomes of
    treated and untreated observations (*sum_treated* and *sum_untreated*) and
    the number of treated and untreated observations (*n_treated* and
    *n_untreated*).

    Args:
        sum_treated and sum2_treated (float): sum of outcomes of treatment individuals.
        n_treated (int): number of treated individuals.
        sum_untreated and sum2_untreated (float): sum of outcomes of untreated individuals.
        n_untreated (int): number of untreated individuals.

    Returns:
        out (float): the estimated treatment effect

    Example:
    >>> sum_t, n_t = 100, 10.0
    >>> sum_unt, n_unt = 1000, 20.0
    >>> _compute_treatment_effect_raw(sum_t, n_t, sum_unt, n_unt)
    -40.0

    """
    out1 = sum_treated / n_treated - sum_untreated / n_untreated
    out2 = sum2_treated / n_treated - sum2_untreated / n_untreated

    return out1, out2


def predict_causaltree(ctree, X):
    """Predicts individual treatment effects for a causal tree.

    Predicts individual treatment effects for new observed features *x*
    on a fitted causal tree *ctree*.

    Args:
        ctree (pd.DataFrame): fitted causal tree represented in a pd.DataFrame
        X (np.array): data on new observations

    Returns:
        predictions (np.array): treatment predictions.

    """
    n = len(X)
    predictions1 = np.empty((n,))
    predictions2 = np.empty((n,))

    for i, row in enumerate(X):
        predictions1[i], predictions2[i]  = _predict_row_causaltree(ctree, row)
    
    predictions = np.array((predictions1, predictions2))

    return predictions


def _predict_row_causaltree(ctree, row):
    """Predicts treatment effect for a single individual.

    Predicts individual treatment effects for new observed features *row* for a
    single individual on a fitted causal tree *ctree*.

    Args:
        ctree (pd.DataFrame): fitted causal tree represented in a pd.DataFrame
        row (np.array): 1d array of features for single new observation

    Returns:
        prediction (float): treatment prediction.

    """
    current_id = 0
    while np.isnan(ctree.loc[current_id, "treat_effect_1"]):
        split_feat = ctree.loc[current_id, "split_feat"]
        go_left = row[split_feat] <= ctree.loc[current_id, "split_value"]

        if go_left:
            current_id = ctree.loc[current_id, "left_child"]
        else:
            current_id = ctree.loc[current_id, "right_child"]

    return ctree.loc[current_id, "treat_effect_1"], ctree.loc[current_id, "treat_effect_2"]
