from nn_networks import FullyConnectedNN
from nn_metrics import ClassificationMetrics
from dataset import torch_load_dataset

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.utils.tensorboard as tb
import torch.nn as nn

import os


# This class takes as argument an instance of nn.Module
# and provides useful methods to perform all operations
# needed in the experiment.
class NeuralNetworkRunner:

    def __init__(self, model, train_data=None, val_data=None, test_data=None, loss_fn=nn.CrossEntropyLoss(), optimizer=None, tensorboard=None):
        self.model = model
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

    # Perform validation
    def _validate(self):
        self.model.eval()
        self.metrics.clear()

        # Pytorch shall not memorize the gradients of this calculation
        with torch.no_grad():
            for (X, yt) in self.val_data:
                # Forward pass
                Y = self.model(X)

                # The index of the largest output along the second dimension gives the predicted
                # class label
                y = Y.argmax(-1)

                # Track evaluation metrics
                self.metrics.add(y, yt)

    # Perform training
    # Pass as arguments the desired number of epochs
    def train(self, epochs=25, lr_scheduler=lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.5)):

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

            # Update the learning rate according to the scheduler
            lr_scheduler.step()

        print("Final validation mAcc: ", self.metrics.mAcc())

    def get_optimizer(self):
        return self.optimizer

    def get_metrics(self):
        return self.metrics


models = [FullyConnectedNN([84, 42, 21], 4), FullyConnectedNN(
    [84, 32, 10], 4), FullyConnectedNN([84, 32, 10], 4, activation_type=nn.Sigmoid)]

schedulers = [lambda o: optim.lr_scheduler.MultiStepLR(o, [25, 40, 50], gamma=0.1), lambda o: optim.lr_scheduler.MultiStepLR(
    o, [10, 15], gamma=0.5), lambda o: optim.lr_scheduler.ReduceLROnPlateau(o)]

experiments = [[model, scheduler]
               for model in models for scheduler in schedulers]

for idx, (model, scheduler) in enumerate(experiments):
    tensorboard = tb.SummaryWriter(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "..", "tb_logs", "nn_{}".format(idx)))

    runner = NeuralNetworkRunner(model, tensorboard=tensorboard)
    runner.train(epochs=55, lr_scheduler=scheduler)

    # , labels = ["normal", "bacteria", "virus", "covid"]
    # runner.get_metrics().plot_confusion_matrix(tensorboard=tensorboard)
