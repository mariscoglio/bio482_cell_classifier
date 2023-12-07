from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings(
    "ignore",
    message="Lazy modules are a new feature under heavy development so changes to the API or functionality can happen at any moment.",
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, TensorDataset


class SkTorchEstimator(nn.Module, ABC):
    def __init__(
        self,
        num_epochs: int,
        loss_function: nn.Module,
        optimizer_class: torch.optim.Optimizer,
        batch_size: int,
        lr: float,
        save_train_loss: bool = False,
    ) -> None:
        """Pytorch and Sklearn compatible class implementing Utime

        Args:
            num_epochs (int): Number of epochs to train.
            loss_function (nn.Module): The class of the loss function to use.
            optimizer_class (torch.optim.Optimizer): The optimize class to use.
            lr (float): The learning rate of the optimizer
            save_train_loss (bool, optional): Whether to save the train loss for every epoch during training. Defaults to False.
            in_dim (int, optional): The number of in channels. Defaults to 5.
            out_dim (int, optional): The number of outputs desired. Defaults to 1.
        """
        # print("new init !!!")
        super().__init__()
        self.num_epochs = num_epochs
        self.lr = lr
        self.optimizer_class = optimizer_class
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.save_train_loss = save_train_loss

        # TODO: add random state and reproducibility code if using dataloaders

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _reset_weights(self) -> None:
        """Reset model weights to avoid weight leakage."""
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, X_val=None, y_val=None):
        """Sklearn compatibility. Train the neural net on X and y.

        Args:
            X (torch.Tensor): The predictors
            y (torch.Tensor): The response
        """
        self._reset_weights()
        train_loader = DataLoader(
            dataset=TensorDataset(
                torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32),
                # torch.tensor(y.to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
                torch.tensor(y.to_numpy(dtype=np.float32), dtype=torch.int64),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        if self.save_train_loss:
            self.training_loss = np.zeros((self.num_epochs,))
        if X_val is not None and y_val is not None:
            self.validation_loss = np.zeros((self.num_epochs,))

        self.train()
        for epoch in tqdm.tqdm(range(self.num_epochs)):
            if self.save_train_loss:
                # Set current loss value
                current_train_loss = 0.0
            # Iterate over the DataLoader for training data
            for i, (inputs, targets) in enumerate(train_loader):
                # Zero the gradients
                self.optimizer.zero_grad()
                # Perform forward pass
                outputs = self(inputs)

                # Compute loss
                loss = self.loss_function(outputs, targets)
                # Perform backward pass
                loss.backward()
                # Perform optimization
                self.optimizer.step()
                if self.save_train_loss:
                    current_train_loss += loss.item()  # / inputs.shape[0]

            # self.optimizer.zero_grad()
            # y_preds = self(X)
            # loss = self.loss_function(y_preds, y)
            # loss.backward()
            # self.optimizer.step()

            if self.save_train_loss:
                current_train_loss /= len(train_loader)
                self.training_loss[epoch] = current_train_loss
                # self.training_loss[epoch] = loss.item()

            if X_val is not None and y_val is not None:
                self.validation_loss[epoch] = self.score(X_val, y_val)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn compatibility. Output the prediction in numpy format

        Args:
            X (torch.Tensor): The data to predict on.

        Returns:
            np.ndarray: The prediction
        """
        with torch.no_grad():
            output = self(torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32))
            pred = output.argmax(dim=1, keepdim=True)
            return pred.numpy()

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """Sklearn compatibility. Score the output of the model using the loss.

        Args:
            X (torch.Tensor): The data to socre one
            y (torch.Tensor): The reponse to use as ground truth

        Returns:
            float: A value for the score.
        """
        # TODO:
        # test_loader = DataLoader(
        #     dataset=TensorDataset(X, y),
        #     batch_size=self.batch_size,
        #     shuffle=False,
        # )
        test_loss = 0.0
        self.eval()
        with torch.no_grad():
            # TODO:
            # # Iterate over the test data and generate predictions
            # for i, (inputs, targets) in enumerate(test_loader):
            #     outputs = self(inputs)
            #     loss = self.loss_function(outputs, targets)
            #     test_loss += loss.item() #/ inputs.shape[0]
            # # test_loss /= len(test_loader)
            preds = self(torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32))
            test_loss = self.loss_function(
                torch.tensor(y.to_numpy(dtype=np.float32), dtype=torch.float32), preds
            ).item()
        return test_loss

    def set_params(self, **parameters):
        """Sklearn compatibility. Set the parameters of the model."""
        for parameter, value in parameters.items():
            if hasattr(self, parameter):
                setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict:
        """Sklearn compatibility. Get the parameters of the model.

        Args:
            deep (bool, optional): Only for compatibility. Defaults to True.

        Returns:
            dict: Important parameters of the model.
        """
        return {
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "optimizer_class": self.optimizer_class,
            "loss_function": self.loss_function,
            "save_train_loss": self.save_train_loss,
        }


class FeedForwardExample(SkTorchEstimator):
    def __init__(
        self,
        num_epochs: int,
        optimizer_class: torch.optim.Optimizer,
        loss_function: nn.Module,
        batch_size: int,
        lr: float,
        save_train_loss: bool = False,
        num_classes: int = 4,
    ) -> None:
        """Pytorch and Sklearn compatible class implementing a simple feedforward 2 layered neural net

        Args:
            num_epochs (int): Number of epochs to train.
            optimizer_class (torch.optim.Optimizer): The optimize class to use.
            loss_function (nn.Module): The class of the loss function to use.
            lr (float): The learning rate of the optimizer
            save_train_loss (bool, optional): Whether to save the train loss for every epoch during training. Defaults to False.
            num_classes (int, optional): The number of outputs desired. Defaults to 4.
        """
        super().__init__(
            num_epochs=num_epochs,
            loss_function=loss_function,
            optimizer_class=optimizer_class,
            batch_size=batch_size,
            lr=lr,
            save_train_loss=save_train_loss,
        )
        self.layers = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(num_classes),
        )
        
        self.optimizer = self.optimizer_class(params=self.parameters(), lr=self.lr)

    def _reset_weights(self):
        """
        Try resetting model weights to avoid weight leakage.
        """
        for layer in self.layers.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Pytorch compatibility. Do a forward pass in the neural net.

        Args:
            X (torch.Tensor): The data to pass forward.

        Returns:
            torch.Tensor: The output of the neural net.
        """
        return self.layers(X)