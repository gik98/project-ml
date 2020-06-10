import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from nn_networks import ConvNN, AugmentedConvNN
from nn_operations import NeuralNetworkRunner
from dataset import torch_load_dataset_augmented, weight_vector

train_transforms = transforms.Compose(
    [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

tensorboard = tb.SummaryWriter(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "tb_logs", "cnn_{}".format(0)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_deep():
    train_data = datasets.ImageFolder(
        "dataset/images/train", transform=train_transforms)
    test_data = datasets.ImageFolder(
        "dataset/images/val", transform=train_transforms)
    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)
    val_dataset = torch.utils.data.DataLoader(test_data, batch_size=64)

    model = ConvNN().to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.05)
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    scheduler = {"scheduler": lambda o: optim.lr_scheduler.MultiStepLR(
        o, [30, 50, 60], gamma=0.1), "epoch": 70}

    runner = NeuralNetworkRunner(
        model, loss_fn=criterion, optimizer=optimizer, tensorboard=tensorboard, train_data=train_dataset, val_data=val_dataset)
    runner.train(lr_setup=scheduler)
    runner.get_metrics().plot_confusion_matrix(tensorboard=tensorboard, labels=[
        "normal", "bacteria", "virus", "covid"], tag="cnn_{}".format(0))


def test_deep_augmented(store_dir=None, test_model=False, test_dir=None):
    train_dataset = torch_load_dataset_augmented(
        name="train", batch_size=64, shuffle=True)
    val_dataset = torch_load_dataset_augmented(
        name="val", batch_size=64, shuffle=False)
    test_dataset = torch_load_dataset_augmented(name="test", batch_size=64, shuffle=False)

    model = AugmentedConvNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_vector)
    #optimizer = optim.SGD(model.parameters(), lr=0.001) #0.05
    optimizer = optim.Adagrad(model.parameters(), lr=0.001) #weight_decay=0.1

    scheduler = {"scheduler": lambda o: optim.lr_scheduler.MultiStepLR(
        o, [25, 35, 45], gamma=0.1), "epoch": 50}
    #scheduler = {"scheduler": lambda o: optim.lr_scheduler.ReduceLROnPlateau(o, "max", verbose=True, patience=8), "epoch": 90}
    runner = NeuralNetworkRunner(
        model, loss_fn=criterion, optimizer=optimizer, tensorboard=tensorboard, train_data=train_dataset, val_data=val_dataset)
    runner.train(lr_setup=scheduler)
    runner.get_metrics().plot_confusion_matrix(tensorboard=tensorboard, labels=[
        "normal", "bacteria", "virus", "covid"], tag="cnn_{}".format(0))

test_deep()

#parser = argparse.ArgumentParser(description="Neural Network experiments")
#
#parser.add_argument("--load", "-l", dest="load_path",
#                    help="Load the trained model from a serialized file. It implies the -t flag.")
#parser.add_argument("--save", "-s", dest="save_path",
#                    help="Store trained models to the specified directory", default="models")
#parser.add_argument("--test", "-t", dest="test",
#                    action="store_true", help="Output test data files")
#
#args = parser.parse_args()
#
#test_dir = os.path.join(os.getcwd(), "output")
#if args.load_path:
#    from_model(os.path.join(os.getcwd(), args.load_path), test_dir=test_dir)
#else:
#    test_deep_augmented(store_dir=os.path.join(
#        os.getcwd(), args.save_path), test_model=args.test, test_dir=test_dir)
#