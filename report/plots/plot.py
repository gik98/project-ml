import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

import csv

mpl.use("Agg")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)
def exp_smooth(y, weight):
    last = y[0]
    smoothed = list()
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

def plot_network(mAcc_train, mAcc_val, output_mAcc, loss, output_loss):
    x_acc = []
    y_acc_train = []
    y_acc_val = []

    with open(mAcc_train, "r") as f:
        parsed = csv.reader(f, delimiter=",")
        for line in parsed:
            x_acc.append(float(line[1]))
            y_acc_train.append(float(line[2]))

    with open(mAcc_val, "r") as f:
        parsed = csv.reader(f, delimiter=",")
        for line in parsed:
            y_acc_val.append(float(line[2]))

    plt.figure()
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.15, left=0.15)

    plt.plot(x_acc, y_acc_train, label="Training")
    plt.plot(x_acc, y_acc_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    #plt.title("Learning: nn")

    plt.savefig(output_mAcc)

    ####

    x_loss = []
    y_loss = []

    with open(loss, "r") as f:
        parsed = csv.reader(f, delimiter=",")
        for line in parsed:
            x_loss.append(float(line[1]))
            y_loss.append(float(line[2]))

    y_loss_smooth = exp_smooth(y_loss, 0.9)

    plt.figure()
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.15, left=0.15)

    plt.plot(x_loss, y_loss, label="Training loss")
    plt.plot(x_loss, y_loss_smooth, label="Training loss (EWMA 0.9)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    #plt.title("Learning: nn")

    plt.savefig(output_loss)

plot_network("nn-mAcc-train.csv", "nn-mAcc-val.csv", "plot_nn_macc.pdf", "nn-loss.csv", "plot_nn_loss.pdf")
plot_network("dnn-mAcc-train.csv", "dnn-mAcc-val.csv", "plot_dnn_macc.pdf", "dnn-loss.csv", "plot_dnn_loss.pdf")
