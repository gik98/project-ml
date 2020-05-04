import numpy as np
import os

train_size = 3569
validation_size = 1189
test_size = 1191

feature_cnt = 84

FEAT_NORMAL = 1
FEAT_BACTERIA = 2
FEAT_VIRUS = 3
FEAT_COVID = 4

base_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "dataset")


def load_dataset(name="test"):
    dataset_size = -1
    if name == "train":
        dataset_size = train_size
    elif name == "val":
        dataset_size = validation_size
    elif name == "test":
        dataset_size = test_size

    if dataset_size == -1:
        return

    features = np.zeros((dataset_size, feature_cnt))
    labels = np.zeros(dataset_size)

    for i in range(dataset_size):
        features[i, :] = np.load(os.path.join(
            base_path, "features", name, "{:04}.npy".format(i)))

    if name != "test":
        with open(os.path.join(base_path, "labels", "{}_labels.txt".format(name))) as file:
            for line in file:
                row, lbl = line.strip().split()
                row = int(row)

                if lbl == "normal":
                    labels[row] = FEAT_NORMAL
                elif lbl == "bacteria":
                    labels[row] = FEAT_BACTERIA
                elif lbl == "viral":
                    labels[row] = FEAT_VIRUS
                elif lbl == "COVID-19":
                    labels[row] = FEAT_COVID

    return features, labels

def evaluate(predictions, truth, output = True):
    # {class} * {success, fails}
    stats = np.zeros((5, 2))

    for i in range(len(predictions)):
        if predictions[i] == truth[i]:
            stats[int(truth[i])][0] += 1
        else:
            stats[int(truth[i])][1] += 1

    accuracy = 0
    for i in range(1, 5):
        accuracy += stats[i][0] / (stats[i][0] + stats[i][1])
        if output:
            print("Class {}: success rate {}/{} ({}%)".format(i,
                                                        stats[i][0], stats[i][0] + stats[i][1], stats[i][0] / (stats[i][0] + stats[i][1]) * 100))

    success = np.sum(stats[:, 0])
    errors = np.sum(stats[:, 1])

    if output:
        print("Overall success rate: {}/{} ({}%)".format(success,
                                                        success + errors, success / (success + errors) * 100))
        print("Interclass success rate: {}%".format(accuracy * 100/4))

    return accuracy * 100 / 4