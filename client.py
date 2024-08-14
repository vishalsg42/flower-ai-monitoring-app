import argparse
import warnings
from collections import OrderedDict

import datasets
import flwr as fl
import torch
from torch.utils.data import DataLoader
from data_loader import DataClientLoader
from model_loader import ModelLoader  # Import the ModelLoader class

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
        # if model_str == "alexnet":
        #     print("Using AlexNet")
        #     self.model = utils.load_alexnet(classes=10)
        # else:
        #     print("Using EfficientNet")
        #     self.model = utils.load_efficientnet(classes=10)

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

        train_valid = self.trainset.train_test_split(self.validation_split, seed=42)
        trainset = train_valid["train"]
        valset = train_valid["test"]

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size)

        results = utils.train(self.model, train_loader, val_loader, epochs, self.device)

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


# def client_dry_run(device: torch.device = "cpu"):
#     """Weak tests to check whether all client methods are working as expected."""

#     # model = utils.load_efficientnet(classes=10)
#     data_loader = DataClientLoader()
#     model_loader = ModelLoader(model_str="efficientnet")
#     # trainset, testset  = DataClientLoader().load_data(0, False)
#     trainset, testset = data_loader.load_data(0, toy=True)
#     client = CifarClient(trainset, testset, device)
#     client.fit(
#         utils.get_model_params(model_loader.load_model()),
#         {"batch_size": 16, "local_epochs": 1},
#     )

#     client.evaluate(utils.get_model_params(model_loader.load_model()), {"val_steps": 32})

#     print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument(
    #     "--dry",
    #     type=bool,
    #     default=False,
    #     required=False,
    #     help="Do a dry-run to check the client",
    # )
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

    # if args.dry:
    #     client_dry_run(device)
    # else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        # trainset, testset = utils.load_partition(args.client_id)
        # print(f"Client {args.client_id} loaded data partition")

        # if args.toy:
        #     trainset = trainset.select(range(10))
        #     testset = testset.select(range(10))
        
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

# import argparse
# import warnings
# from collections import OrderedDict
# from abc import ABC, abstractmethod
# from .data_loader import DataClientLoader

# import datasets
# import flwr as fl
# import torch
# from torch.utils.data import DataLoader

# import utils

# warnings.filterwarnings("ignore")


# Interface for Model Connection
# class IModelConnection(ABC):
#     @abstractmethod
#     def load_model(self) -> torch.nn.Module:
#         pass


# class AlexNetConnection(IModelConnection):
#     def load_model(self) -> torch.nn.Module:
#         return utils.load_alexnet(classes=10)


# # class EfficientNetConnection(IModelConnection):
# #     def load_model(self) -> torch.nn.Module:
# #         return utils.load_efficientnet(classes=10)


# # # Factory to Create Model Connections
# # class ModelConnectionFactory:
# #     def create_connection(self, model_type: str) -> IModelConnection:
# #         if model_type == "alexnet":
# #             return AlexNetConnection()
# #         elif model_type == "efficientnet":
# #             return EfficientNetConnection()
# #         else:
# #             raise ValueError("Invalid model type")


# # # DataLoaderFactory Interface
# # class DataLoaderFactory(ABC):
# #     @abstractmethod
# #     def load_data(self, client_id: int, toy: bool) -> tuple[datasets.Dataset, datasets.Dataset]:
# #         pass


# # class Cifar10DataLoader(DataLoaderFactory):
# #     def load_data(self, client_id: int, toy: bool) -> tuple[datasets.Dataset, datasets.Dataset]:
# #         trainset, testset = utils.load_partition(client_id)
# #         if toy:
# #             trainset = trainset.select(range(10))
# #             testset = testset.select(range(10))
# #         return trainset, testset


# # Flower Client Implementation
# class CifarClient(fl.client.NumPyClient):
#     def __init__(
#         self,
#         model_connection: IModelConnection,
#         data_loader_factory: DataLoaderFactory,
#         client_id: int,
#         device: torch.device,
#         validation_split: int = 0.1,
#         toy: bool = False,
#     ):
#         self.device = device
#         self.validation_split = validation_split
        
#         # Load data using the factory
#         self.trainset, self.testset = data_loader_factory.load_data(client_id, toy)
        
#         # Load model using the connection factory
#         self.model = model_connection.load_model()

#     def set_parameters(self, parameters):
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         print("Fitting model on client side ...")
#         self.set_parameters(parameters)

#         batch_size: int = config["batch_size"]
#         epochs: int = config["local_epochs"]

#         train_valid = self.trainset.train_test_split(self.validation_split, seed=42)
#         trainset = train_valid["train"]
#         valset = train_valid["test"]

#         train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(valset, batch_size=batch_size)

#         results = utils.train(self.model, train_loader, val_loader, epochs, self.device)

#         parameters_prime = utils.get_model_params(self.model)
#         num_examples_train = len(trainset)

#         return parameters_prime, num_examples_train, results

#     def evaluate(self, parameters, config):
#         print("Evaluating model on client side ...")
#         self.set_parameters(parameters)

#         steps: int = config["val_steps"]
#         testloader = DataLoader(self.testset, batch_size=16)

#         loss, accuracy = utils.test(self.model, testloader, steps, self.device)
#         return float(loss), len(self.testset), {"accuracy": float(accuracy)}


# def client_dry_run(device: torch.device = "cpu"):
#     model_connection = EfficientNetConnection()
#     data_loader_factory = Cifar10DataLoader()

#     model = model_connection.load_model()
#     trainset, testset = data_loader_factory.load_data(0, toy=True)
#     trainset = trainset.select(range(10))
#     testset = testset.select(range(10))

#     client = CifarClient(model_connection, data_loader_factory, 0, device, toy=True)
#     client.fit(
#         utils.get_model_params(model),
#         {"batch_size": 16, "local_epochs": 1},
#     )

#     client.evaluate(utils.get_model_params(model), {"val_steps": 32})

#     print("Dry Run Successful")


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Flower")
#     parser.add_argument(
#         "--dry",
#         type=bool,
#         default=False,
#         required=False,
#         help="Do a dry-run to check the client",
#     )
#     parser.add_argument(
#         "--client-id",
#         type=int,
#         default=0,
#         choices=range(0, 10),
#         required=False,
#         help="Specifies the artificial data partition of CIFAR10 to be used. \
#         Picks partition 0 by default",
#     )
#     parser.add_argument(
#         "--toy",
#         action="store_true",
#         help="Set to true to quickly run the client using only 10 datasamples. \
#         Useful for testing purposes. Default: False",
#     )
#     parser.add_argument(
#         "--use_cuda",
#         type=bool,
#         default=False,
#         required=False,
#         help="Set to true to use GPU. Default: False",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="efficientnet",
#         choices=["efficientnet", "alexnet"],
#         help="Use either Efficientnet or Alexnet models. \
#              If you want to achieve differential privacy, please use the Alexnet model",
#     )

#     args = parser.parse_args()

#     device = torch.device(
#         "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
#     )

#     if args.dry:
#         client_dry_run(device)
#     else:
#         model_connection_factory = DataClientLoader()
#         # model_connection = model_connection_factory.create_connection(args.model)
        
#         # data_loader_factory = Cifar10DataLoader()

#         client = CifarClient(model_connection_factory, data_loader_factory, args.client_id, device, toy=args.toy)
#         print(f"Client {args.client_id} started")
#         # TODO update server address & port number from the environment variable 
#         fl.client.start_client(server_address="3.95.62.233:8080", client=client)


# if __name__ == "__main__":
#     main()
