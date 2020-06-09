import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from PIL import Image
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ClassificationMetrics:
    # Constructor takes the number of classes
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        # Initialize a confusion matrix
        self.C = torch.zeros(num_classes, num_classes).to(device)

    # Update the confusion matrix with the new scores
    def add(self, yp, yt):
        # yp: 1D tensor with predictions
        # yt: 1D tensor with ground-truth targets
        with torch.no_grad():  # We require no computation graph
            self.C += (yt*self.C.shape[1]+yp).bincount(
                minlength=self.C.numel()).view(self.C.shape).float()

    def clear(self):
        # We set the confusion matrix to zero
        self.C.zero_()

    # Computes the global accuracy
    def acc(self):
        return self.C.diag().sum().item()/self.C.sum()

    # Computes the class-averaged accuracy
    def mAcc(self):
        return (self.C.diag()/self.C.sum(-1)).mean().item()

    # Computers the class-averaged Intersection over Union
    def mIoU(self):
        return (self.C.diag()/(self.C.sum(0)+self.C.sum(1)-self.C.diag())).mean().item()

    # Returns the confusion matrix
    def confusion_matrix(self):
        return self.C

    def _plot_to_image(self):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = Image.open(buf)
        #image = tf.image.decode_png(buf.getvalue(), channels=4)
        #image = tf.expand_dims(image, 0)

        image_np = np.array(image)
        return image_np[...,:3]

    def plot_confusion_matrix(self, tensorboard, labels=None, tag=""):
        if labels == None:
            labels = range(1, self.num_classes + 1)
        df_cm = pd.DataFrame(self.C.cpu().numpy(), labels, labels)
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

        tensorboard.add_image("Confusion matrix {}".format(tag), self._plot_to_image(), dataformats="HWC")

