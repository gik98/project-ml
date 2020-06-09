import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from nn_networks import ConvNN, AugmentedConvNN
from nn_operations import NeuralNetworkRunner
from dataset import torch_load_dataset

train_transforms = transforms.Compose(
    [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])


train_data = datasets.ImageFolder(
    "dataset/images/train", transform=train_transforms)
test_data = datasets.ImageFolder(
    "dataset/images/val", transform=train_transforms)

train_dataset = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True)
val_dataset = torch.utils.data.DataLoader(test_data, batch_size=64)

#model = ConvNN()
model = AugmentedConvNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)
scheduler = {"scheduler": lambda o: optim.lr_scheduler.MultiStepLR(
    o, [30, 50, 60], gamma=0.1), "epoch": 70}


#
#classes = ["normal", "bacteria", "viral", "COVID-19"]
#
# for epoch in range(100):  # loop over the dataset multiple times
#
#    running_loss = 0.0
#    for i, data in enumerate(train_dataset, 0):
#        # get the inputs; data is a list of [inputs, labels]
#        inputs, labels = data
#
#        # zero the parameter gradients
#        optimizer.zero_grad()
#
#        # forward + backward + optimize
#        outputs = model(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#
#        running_loss += loss.item()
#
#    # print statistics
#    print('[%d, %5d] loss: %.3f' %
#          (epoch + 1, i + 1, running_loss / len(train_dataset)))
#    running_loss = 0.0
#    class_correct = list(0. for i in range(4))
#    class_total = list(0. for i in range(4))
#    with torch.no_grad():
#        for data in val_dataset:
#            images, labels = data
#            outputs = model(images)
#            _, predicted = torch.max(outputs, 1)
#            c = (predicted == labels).squeeze()
#            for i in range(4):
#                label = labels[i]
#                class_correct[label] += c[i].item()
#                class_total[label] += 1
#    for i in range(4):
#        print('Accuracy of {}: {}/{}'.format(classes[i], class_correct[i], class_total[i]))
#
#print('Finished Training')
#

tensorboard = tb.SummaryWriter(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "tb_logs", "cnn_{}".format(0)))
# runner = NeuralNetworkRunner(model, optimizer=optimizer, tensorboard=tensorboard,
#                             train_data=train_dataset, val_data=val_dataset)
runner = NeuralNetworkRunner(model, augmented=True, optimizer=optimizer, tensorboard=tensorboard, train_data=(train_dataset, torch_load_dataset(
    "train", shuffle=True)), val_data=(val_dataset, torch_load_dataset("val", shuffle=False)))

runner.train(lr_setup=scheduler)
runner.get_metrics().plot_confusion_matrix(tensorboard=tensorboard, labels=[
    "normal", "bacteria", "virus", "covid"], tag="cnn_{}".format(0))


# model.to(device)
#
#epochs = 10
#steps = 0
#running_loss = 0
#print_every = 10
#train_losses, val_losses = [], []
#
# for epoch in range(epochs):
#    for inputs, labels in train_dataset:
#        steps += 1
#        inputs, labels = inputs.to(device), labels.to(device)
#        optimizer.zero_grad()
#        logps = model.forward(inputs)
#        loss = criterion(logps, labels)
#        loss.backward()
#        optimizer.step()
#        running_loss += loss.item()
#
#    test_loss = 0
#    accuracy = 0
#    model.eval()
#    with torch.no_grad():
#        for inputs, labels in val_dataset:
#            inputs, labels = inputs.to(device), labels.to(device)
#            logps = model.forward(inputs)
#            batch_loss = criterion(logps, labels)
#            test_loss += batch_loss.item()
#            ps = torch.exp(logps)
#            top_p, top_class = ps.topk(1, dim=1)
#            equals = top_class == labels.view(*top_class.shape)
#            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
#    train_losses.append(running_loss/len(train_dataset))
#    val_losses.append(test_loss/len(val_dataset))
#    print(f"Epoch {epoch+1}/{epochs}.. "
#          f"Train loss: {running_loss/len(train_dataset):.3f}.. "
#          f"Test loss: {test_loss/len(val_dataset):.3f}.. "
#          f"Test accuracy: {accuracy/len(val_dataset):.3f}")
#    running_loss = 0
#    model.train()
#
##torch.save(model, 'aerialmodel.pth')
#
#plt.plot(train_losses, label='Training loss')
#plt.plot(val_losses, label='Validation loss')
# plt.legend(frameon=False)
# plt.show()
