params = {
    'iterations': trial.suggest_int('iterations', 700, 2000),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
    'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
    'bootstrap_type': trial.suggest_categorical(
        'bootstrap_type',
        ['Bayesian', 'Bernoulli', 'MVS'],
    ),
    'grow_policy': trial.suggest_categorical(
        'grow_policy',
        ['SymmetricTree', 'Depthwise', 'Lossguide'],
    ),
    'border_count': trial.suggest_int('border_count', 64, 254),
}

if params['bootstrap_type'] == 'Bayesian':
    params['bagging_temperature'] = trial.suggest_float(
        'bagging_temperature',
        0.0,
        10.0,
    )
else:
    params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)

if params['grow_policy'] == 'SymmetricTree':
    params['depth'] = trial.suggest_int('depth', 4, 10)
    params['boosting_type'] = trial.suggest_categorical(
        'boosting_type',
        ['Ordered', 'Plain'],
    )

elif params['grow_policy'] == 'Depthwise':
    params['depth'] = trial.suggest_int('depth', 6, 12)
    params['min_data_in_leaf'] = trial.suggest_int(
        'min_data_in_leaf',
        100,
        5000,
        log=True,
    )

else:
    params['depth'] = trial.suggest_int('depth', 6, 14)
    params['max_leaves'] = trial.suggest_int('max_leaves', 32, 512, log=True)
    params['min_data_in_leaf'] = trial.suggest_int(
        'min_data_in_leaf',
        100,
        5000,
        log=True,
    )