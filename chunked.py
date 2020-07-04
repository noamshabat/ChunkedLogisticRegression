import pandas as pd
import os
import pathlib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
import time
import DataSplitter

startT = int(round(time.time() * 1000))


curr_path = os.path.dirname(os.path.realpath(__file__))
file_path = pathlib.Path(__file__).parent.absolute()

print(file_path)

partial_fit = True
chunk_size = 100
max_iter = 12
alpha = 0.03
penalty = 'l2'
loss='log'

totalCount = 0

x_test = None
y_test = None
col_names = None
feature_cols = None

# instantiate the model (using the default parameters)
logreg = SGDClassifier(max_iter=max_iter, alpha=alpha, penalty=penalty, loss=loss)
scaler = MinMaxScaler(feature_range=(0, 1))

shuffle = DataSplitter.get_split_data('./tmp')

for chunks in shuffle():
    for chunk in chunks():
        if col_names is None:
            col_names = chunk.columns.values
            feature_cols = col_names[col_names != 'label']  # we want the array to contain only feature cols..
        scaler.partial_fit(chunk[feature_cols])
    break

# in sgd with partial_fit we need to iterate on the data ourselves.
for i in range((max_iter if partial_fit else 1)):
    shuffles = DataSplitter.get_split_data('./tmp')
    for chunks in shuffles():
        # if i%10 == 0:
        print(f"passed {i} iterations")

        for chunk in chunks():
            chunk = chunk.sample(frac=1).reset_index(drop=True)
            totalCount += len(chunk.index)

            X = scaler.transform(chunk[feature_cols])  # chunk[feature_cols]  # Features
            X[X > 1] = 1
            X[X < 0] = 0
            y = chunk.label  # Target variable

            if partial_fit:
                logreg.partial_fit(X, y, [0, 1])
            else:
                logreg.fit(X, y)


print('total count', totalCount)

test_data = DataSplitter.get_test_data('./tmp')
x_test = scaler.transform(test_data[feature_cols])
y_test = test_data.label
#
y_pred = logreg.predict(x_test)

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# plt.show()
# Text(0.5,257.44,'Predicted label')
#TODO: why is text undefined?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print('F1 score:', metrics.f1_score(y_test, y_pred))

endT = int(round(time.time() * 1000))
print("runtime: ", endT - startT)