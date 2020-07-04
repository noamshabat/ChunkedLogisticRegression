from sklearn.preprocessing import MinMaxScaler
import DataSplitter


def run_learner(learner, max_iter, data_folder, learner_args):
    total_count = 0

    col_names = None
    feature_cols = None

    # instantiate the model (using the default parameters)
    logreg = learner(max_iter=max_iter, **learner_args)
    scaler = MinMaxScaler(feature_range=(0, 1))

    shuffle = DataSplitter.get_split_data(data_folder)

    for chunks in shuffle():
        for chunk in chunks():
            if col_names is None:
                col_names = chunk.columns.values
                feature_cols = col_names[col_names != 'label']  # we want the array to contain only feature cols..
            scaler.partial_fit(chunk[feature_cols])
        break

    # in sgd with partial_fit we need to iterate on the data ourselves.
    for i in range(max_iter):
        shuffles = DataSplitter.get_split_data(data_folder)
        for chunks in shuffles():
            for chunk in chunks():
                chunk = chunk.sample(frac=1).reset_index(drop=True)
                total_count += len(chunk.index)

                X = scaler.transform(chunk[feature_cols])  # chunk[feature_cols]  # Features
                X[X > 1] = 1
                X[X < 0] = 0
                y = chunk.label  # Target variable

                logreg.partial_fit(X, y, [0, 1])

    test_data = DataSplitter.get_test_data(data_folder)
    x_test = scaler.transform(test_data[feature_cols])
    y_test = test_data.label
    #
    y_pred = logreg.predict(x_test)

    # import the metrics class
    from sklearn import metrics
    confusion =  metrics.confusion_matrix(y_test, y_pred)
    result = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "f1": metrics.f1_score(y_test, y_pred),
        "true-negative": confusion[0][0],
        "false-negative": confusion[1][0],
        "false-positive": confusion[0][1],
        "true-positive": confusion[1][1],
    }
    return result
