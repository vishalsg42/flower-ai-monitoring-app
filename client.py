import argparse
import warnings
from collections import OrderedDict

import datasets
import flwr as fl
import torch
from torch.utils.data import DataLoader
from data_loader import DataClientLoader
from model_loader import ModelLoader

import utils

warnings.filterwarnings("ignore")


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: datasets.Dataset,
        testset: datasets.Dataset,
        device: torch.device,
        model_loader: ModelLoader,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split
        self.model = model_loader.load_model()

    def set_parameters(self, parameters):
        """Loads a alexnet or efficientnet model and replaces it parameters with the
        ones given."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        print("Fitting model on client side ...")

        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        train_valid = self.trainset.train_test_split(
            self.validation_split, seed=42)
        trainset = train_valid["train"]
        valset = train_valid["test"]

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size)

        results = utils.train(self.model, train_loader,
                              val_loader, epochs, self.device)

        parameters_prime = utils.get_model_params(self.model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        print("Evaluating model on client side ...")
        # Update local model parameters
        self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=16)

        loss, accuracy = utils.test(self.model, testloader, steps, self.device)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. \
             If you want to achieve differential privacy, please use the Alexnet model",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    data_loader = DataClientLoader()
    trainset, testset = data_loader.load_data(args.client_id, toy=args.toy)
    print(f"Client {args.client_id} loaded data partition")

    # Load model using the ModelLoader
    model_loader = ModelLoader(model_str=args.model)

    # Start Flower client
    client = CifarClient(trainset, testset, device, model_loader).to_client()
    print(f"Client {args.client_id} started")
    fl.client.start_client(server_address="3.95.62.233:8080", client=client)


if __name__ == "__main__":
    main()
