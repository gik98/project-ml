import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_size = 3569
validation_size = 1189
test_size = 1191

feature_cnt = 84
batch_size_default = 32

#weight_vector = torch.FloatTensor([3569.0/951.0, 3569.0/1664.0, 3569.0/909.0, 3569.0/45.0]).to(device)
#weight_vector *= weight_vector
weight_vector = torch.FloatTensor([1e3/951.0, 1e3/1664.0, 1e3/909.0, 1e3/45.0]).to(device)
weight_vector *= weight_vector

feat_string_mapping = ["normal", "bacteria", "viral", "COVID-19"]
string_feat_mapping = {"normal": 0, "bacteria": 1, "viral": 2, "COVID-19": 3}

base_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "dataset")


def get_dataset_size(name):
    dataset_size = -1
    if name == "train":
        dataset_size = train_size
    elif name == "val":
        dataset_size = validation_size
    elif name == "test":
        dataset_size = test_size

    return dataset_size


def load_dataset(name="test", normalize=False):
    dataset_size = get_dataset_size(name)

    if dataset_size == -1:
        return

    features = np.zeros((dataset_size, feature_cnt))
    features -= 1
    labels = np.zeros(dataset_size)

    for i in range(dataset_size):
        features[i, :] = np.load(os.path.join(
            base_path, "features", name, "{:04}.npy".format(i)))

    if name != "test":
        with open(os.path.join(base_path, "labels", "{}_labels.txt".format(name))) as file:
            for line in file:
                row, lbl = line.strip().split()
                row = int(row)
                labels[row] = string_feat_mapping[lbl]

    if normalize:
        mean = features.mean()
        stdev = features.std()
        features -= mean
        features /= stdev

    return features, labels


def torch_load_dataset(name="train", batch_size=batch_size_default, shuffle=True):
    feat, lbl = load_dataset(name=name, normalize=True)

    dataset = data.TensorDataset(torch.from_numpy(
        feat).float(), torch.from_numpy(lbl).long())
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def torch_load_dataset_augmented(name="train", batch_size=batch_size_default, shuffle=True):
    # returns extra row with padded feature
    feat, lbl = load_dataset(name=name, normalize=True)
    feat = torch.from_numpy(feat).float()
    lbl = torch.from_numpy(lbl).long()

    tr = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    imgs = torch.zeros([len(lbl), 256, 256])

    for path in Path("dataset/images/{}".format(name)).rglob("jpeg"):
        n = path.split("/")[-1]
        n = n.split(".")[0]
        imgs[int(n)] = tr(Image.open(path))

    feat = torch.cat((feat, torch.zeros([len(lbl), 256-feature_cnt])), dim=1)
    feat = feat.unsqueeze(2)
    imgs = torch.cat((imgs, feat), dim=2)
    imgs = imgs.unsqueeze(1)

    dataset = data.TensorDataset(imgs, lbl)
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def evaluate(predictions, truth, output=True):
    # {class} * {success, fails}
    stats = np.zeros((4, 2))

    for i in range(len(predictions)):
        if predictions[i] == truth[i]:
            stats[int(truth[i])][0] += 1
        else:
            stats[int(truth[i])][1] += 1

    accuracy = 0
    for i in range(0, 4):
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
