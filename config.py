class Config():

    data_dir = ''
    train_file = 'train.csv'
    test_file = 'test.csv'
    output_prob_file = 'predictions_prob.csv'
    output_label_file = 'predictions_label.csv'

    labels = ['Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5', 'Label_6']

    tfidf_features = 400

    n_parallel = 7

    param = {}
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'auc'
    param['nthread'] = n_parallel

    param['eta'] = 0.1
    param['max_depth'] = 4
    param['min_child_weight'] = 1
    param['subsample'] = 1
    param['colsample_bytree'] = 1
    param['colsample_bylevel'] = 1
    param['lambda'] = 1
    param['alpha'] = 0
    param['seed'] = 0
    param['gamma'] = 0
    param['silent'] = 1

    numround = {'Label_1': 20, 'Label_2': 30, 'Label_3': 30, 'Label_4': 30, 'Label_5': 10, 'Label_6': 10}

    threshold = 0.5
