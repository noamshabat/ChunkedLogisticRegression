from sklearn.linear_model import SGDClassifier

# 1. run data splitter to create pre-shuffled. run just once.
from DataSplitter import create_chunks
from LearningOptimizer import test_all

create = False

if create:
    create_chunks('/home/noam/dev/stockerDailyLogisticModel/daily-2012-01-01-2013-01-01.csv',
              '\t', 1024*1024*50, 10, '/home/noam/dev/data', 0.1)
else:
    config = {
        "SGDClassifier": {
            "learner": SGDClassifier,
            "args": {
                "alpha": [0.001, 0.0001],
                "penalty": ['elasticnet'],
                "loss": ['log'], # , 'perceptron', 'squared_loss', 'huber',
                         #'epsilon_insensitive', 'squared_epsilon_insensitive'],
                "l1_ratio": [0.75, 0.9],
                "warm_start": [True]
                # "class_weight": [None, "balanced"]
            }
        }
    }
    test_all('/home/noam/dev/data', config, '/home/noam/dev/data/optimizer_out.csv')
