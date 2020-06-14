import matplotlib.pyplot as plt
from matplotlib import rc
import csv

rc('text', usetex=True)

x_acc = []
y_acc_train = []
y_acc_val = []

with open("nn-mAcc-train.csv", "r") as f:
    parsed = csv.reader(f, delimiter=",")
    for line in parsed:
        x_acc.append(float(line[1]))
        y_acc_train.append(float(line[2]))

with open("nn-mAcc-val.csv", "r") as f:
    parsed = csv.reader(f, delimiter=",")
    for line in parsed:
        y_acc_val.append(float(line[2]))

plt.plot(x_acc, y_acc_train, label="Training")
plt.plot(x_acc, y_acc_val, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()
