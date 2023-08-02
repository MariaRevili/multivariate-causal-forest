# Multivariate Causal Forest


Credits to timmens: https://github.com/timmens/causal-forest

Minimal Example:
```
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
```
