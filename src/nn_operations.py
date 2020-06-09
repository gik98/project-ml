from nn_networks import FullyConnectedNN, NN_load
from nn_metrics import ClassificationMetrics
from dataset import torch_load_dataset, feat_string_mapping, test_size

import numpy as np

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.utils.tensorboard as tb
import torch.nn as nn

import copy
import os


# This class takes as argument an instance of nn.Module
# and provides useful methods to perform all operations
# needed in the experiment.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA enabled")

class NeuralNetworkRunner:

    def __init__(self, model, train_data=None, val_data=None, test_data=None, loss_fn=nn.CrossEntropyLoss(), optimizer=None, tensorboard=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_data = train_data if train_data != None else torch_load_dataset(
            "train", shuffle=True)
        self.val_data = val_data if val_data != None else torch_load_dataset(
            "val", shuffle=False)
        self.test_data = test_data if test_data != None else torch_load_dataset(
            "test", shuffle=False)
        self.optimizer = optimizer
        if self.optimizer == None:
            self.optimizer = optim.SGD(model.parameters(), lr=0.1,
                                       momentum=0.9, weight_decay=0.000001)

        self.metrics = ClassificationMetrics(model.num_classes)
        self.tensorboard = tensorboard

    # Do one training iteration
    def _train_epoch(self, epoch):
        self.model.train()
        self.metrics.clear()

        # Iterate the batch
        for i, (X, yt) in enumerate(self.train_data):
            X = X.to(device)
            yt = yt.to(device)

            self.optimizer.zero_grad()

            # Forward pass
            Y = self.model(X)

            # Compute the loss, calc prediction
            loss = self.loss_fn(Y, yt)
            y = Y.argmax(-1)

            # Track evaluation metrics + tensorboard if available
            self.metrics.add(y, yt)
            if self.tensorboard:
                self.tensorboard.add_scalar('train/loss', loss.item(),
                                            epoch*len(self.train_data)+i)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
    
    # Perform testing
    def test(self):
        self.model.eval()
        out = []
        with torch.no_grad():
            for idx, (X, yt) in enumerate(self.test_data):
                X = X.to(device)
                yt = yt.to(device)
                
                Y = self.model(X)
                y = Y.argmax(-1)
                out.append(y)
        
        return torch.cat(out)
    
    # Perform validation
    def _validate(self):
        self.model.eval()
        self.metrics.clear()

        # Pytorch shall not memorize the gradients of this calculation
        with torch.no_grad():
            for (X, yt) in self.val_data:
                X = X.to(device)
                yt = yt.to(device)

                # Forward pass
                Y = self.model(X)

                # The index of the largest output along the second dimension gives the predicted
                # class label
                y = Y.argmax(-1)

                # Track evaluation metrics
                self.metrics.add(y, yt)

    # Perform training
    # Pass as arguments the desired number of epochs
    def train(self, stdout_output = True, lr_setup={"scheduler": lambda o: optim.lr_scheduler.MultiStepLR(o, [10, 15], gamma=0.5), "epoch": 25}):
        epochs = lr_setup["epoch"]
        lr_scheduler = lr_setup["scheduler"](self.optimizer)
        for epoch in range(1, epochs+1):
            # Train for one epoch
            self._train_epoch(epoch)

            # Track the metrics in the tensorboard log
            if self.tensorboard:
                self.tensorboard.add_scalar(
                    'train/acc', self.metrics.acc(), epoch)
                self.tensorboard.add_scalar(
                    'train/mAcc', self.metrics.mAcc(), epoch)
                self.tensorboard.add_scalar(
                    'train/mIoU', self.metrics.mIoU(), epoch)
                self.tensorboard.add_scalar(
                    'train/lr', lr_scheduler.get_last_lr()[0], epoch)

            if stdout_output:
                print("Training epoch {}, mAcc: {}".format(epoch, self.metrics.mAcc()))
            # Evaluate the current model
            self._validate()

            # Track the metrics in the tensorboard log
            if self.tensorboard:
                self.tensorboard.add_scalar(
                    'val/acc', self.metrics.acc(), epoch)
                self.tensorboard.add_scalar(
                    'val/mAcc', self.metrics.mAcc(), epoch)
                self.tensorboard.add_scalar(
                    'val/mIoU', self.metrics.mIoU(), epoch)
            if stdout_output:
                print("Validation epoch {}, mAcc: {}".format(epoch, self.metrics.mAcc()))
            # Update the learning rate according to the scheduler
            lr_scheduler.step()

        print("Final validation mAcc: ", self.metrics.mAcc())

    def get_optimizer(self):
        return self.optimizer

    def get_metrics(self):
        return self.metrics

def runner_output_test(runner, output_path):
    with open(output_path, "w") as file:
        l = runner.test()
        for idx, val in enumerate(l):
            file.write("{}\t{}\n".format(idx, feat_string_mapping[val]))


def from_model(load_path=None, test_dir=None):
    tensorboard = tb.SummaryWriter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tb_logs", os.path.basename(load_path)))
    model = NN_load(load_path)
    runner = NeuralNetworkRunner(model, tensorboard=tensorboard)

    output_path = os.path.join(test_dir, "{}.txt".format(os.path.basename(load_path)))
    runner_output_test(runner, output_path)
    runner.get_metrics().plot_confusion_matrix(tensorboard=tensorboard, labels = ["normal", "bacteria", "virus", "covid"], tag=os.path.basename(load_path))