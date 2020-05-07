from dataset import load_dataset, evaluate
from timeutils import measure_time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics

train_feat, train_labels = measure_time(
    load_dataset, label="load dataset (training)", name="train")
val_feat, val_labels = measure_time(
    load_dataset, label="load dataset (validation)", name="val")


#svm_models = [{"model": svm.SVC(C = C, kernel = kernel_type, gamma=gamma), "C": C, "kernel": kernel_type, "gamma": gamma} for kernel_type in ["rbf", "linear", "poly", "sigmoid"] for C in [5e-2, 1e-1, 5e-1, 1, 3, 5, 7, 10] for gamma in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]]
svm_models = [{"model": svm.SVC(C = C, kernel = kernel_type), "C": C, "kernel": kernel_type} for kernel_type in ["rbf", "linear", "poly", "sigmoid"] for C in [5e-2, 1e-1, 5e-1, 1, 3, 5, 7, 10]]

accuracy = np.zeros(len(svm_models))

def validate(idx, obj, show_plot = False):    
    model = obj["model"]
    measure_time(model.fit, "SVM fit", train_feat, train_labels)
    svm_predictions = measure_time(model.predict, "SVM predict", val_feat)
    accuracy[idx] = evaluate(svm_predictions, val_labels, output = False)
    if show_plot:
        metrics.plot_confusion_matrix(model, val_feat, val_labels, normalize="pred")
        plt.show()

#for idx, obj in enumerate(svm_models):
#   validate(idx, obj)

validate(5, svm_models[5], show_plot= True)

idx = np.argmax(accuracy)
print("Model {}: C = {}, kernel type = {}, accuracy {}%".format(idx, svm_models[idx]["C"], svm_models[idx]["kernel"], accuracy[idx]))
