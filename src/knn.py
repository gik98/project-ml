from dataset import load_dataset, evaluate
from timeutils import measure_time

from sklearn import neighbors, metrics
import numpy as np
import matplotlib.pyplot as plt

train_feat, train_labels = measure_time(
    load_dataset, label="load dataset (training)", name="train")
val_feat, val_labels = measure_time(
    load_dataset, label="load dataset (validation)", name="val")


def knn_validate(min_param, max_param, output=False):
    max_param += 1
    metric = np.zeros(max_param)
    for k in range(min_param, max_param):
        knn_model = neighbors.KNeighborsClassifier(n_jobs=-1, n_neighbors=17)

        measure_time(knn_model.fit, "KNN fit", train_feat, train_labels)
        predictions = measure_time(knn_model.predict, "KNN predict", val_feat)

        metric[k] = evaluate(predictions, val_labels, output=output)

        if k == 2:
            metrics.plot_confusion_matrix(knn_model, val_feat, val_labels, normalize="pred")
            plt.show()

    idx = np.argmax(metric)

    return idx, metric[idx], metric

k, val, metric = knn_validate(2, 20)

#plt.plot(metric, 'ro')
#plt.show()

print("K value {}, accuracy {}%".format(k, val))
