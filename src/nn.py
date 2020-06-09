import argparse
import copy
import os

import torch.optim as optim

from nn_operations import from_model
from nn_networks import FullyConnectedNN

parser = argparse.ArgumentParser(description="Neural Network experiments")

parser.add_argument("--load", "-l", dest="load_path",
                    help="Load the trained model from a serialized file. It implies the -t flag.")
parser.add_argument("--save", "-s", dest="save_path",
                    help="Store trained models to the specified directory", default="models")
parser.add_argument("--test", "-t", dest="test",
                    action="store_true", help="Output test data files")

args = parser.parse_args()

test_dir = os.path.join(os.getcwd(), "output")
if args.load_path:
    from_model(os.path.join(os.getcwd(), args.load_path), test_dir=test_dir)
else:
    perform_experiments(store_dir=os.path.join(
        os.getcwd(), args.save_path), test_model=args.test, test_dir=test_dir)


def perform_experiments(store_dir=None, test_model=False, test_dir=None):
    models = [FullyConnectedNN([84, 42, 21], 4), FullyConnectedNN(
        [84, 32, 10], 4), FullyConnectedNN([84, 42, 21, 10], 4), FullyConnectedNN([84, 32, 10], 4, activation_type=nn.Sigmoid)]

    # {"scheduler": lambda o: optim.lr_scheduler.MultiStepLR(o, [25, 40, 50], gamma=0.1), "epoch": 55}
    schedulers = [
        {"scheduler": lambda o: optim.lr_scheduler.MultiStepLR(
            o, [10, 15], gamma=0.5), "epoch": 25},
        {"scheduler": lambda o: optim.lr_scheduler.MultiStepLR(
            o, [20, 30], gamma=0.1), "epoch": 40},
    ]

    optimizers = [
        (lambda model: optim.SGD(model.parameters(),
                                 lr=0.1, momentum=0.9, weight_decay=0.000001)),
        (lambda model: optim.SGD(model.parameters(), lr=0.1)),
        (lambda model: optim.Adagrad(model.parameters(), lr=0.1)),
        (lambda model: optim.Adagrad(
            model.parameters(), lr=0.1, lr_decay=1e-5))
    ]

    experiments = [[copy.deepcopy(model), copy.deepcopy(scheduler), optimizer, "Model #{}, scheduler #{}, optimizer #{}".format(idx_m, idx_s, idx_o)]
                   for idx_m, model in enumerate(models) for idx_s, scheduler in enumerate(schedulers) for idx_o, optimizer in enumerate(optimizers)]

    for idx, (model, scheduler, optimizer_fn, info) in enumerate(experiments):
        tensorboard = tb.SummaryWriter(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "tb_logs", "nn_{}".format(idx)))

        print("Experiment {}, information: {}".format(idx, info))
        runner = NeuralNetworkRunner(
            model, optimizer=optimizer_fn(model), tensorboard=tensorboard)
        runner.train(lr_setup=scheduler)

        runner.get_metrics().plot_confusion_matrix(tensorboard=tensorboard, labels=[
            "normal", "bacteria", "virus", "covid"], tag="nn_{}".format(idx))
        if store_dir is not None:
            store_path = os.path.join(store_dir, "nn_{}.torch".format(idx))
            model.save(store_path)
        if test_model:
            output_path = os.path.join(test_dir, "nn_{}.txt".format(idx))
            runner_output_test(runner, output_path)
