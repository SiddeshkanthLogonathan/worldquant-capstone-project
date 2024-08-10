from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import Sequential
from torch_geometric.utils import to_dense_batch


class MultiPeriodEIIE(nn.Module):
    def __init__(
        self,
        initial_features=3,
        k_size=3,
        conv_mid_features=2,
        conv_final_features=20,
        time_window=50,
        prediction_horizon=1,
        device="cpu",
    ):
        """EIIE (ensemble of identical independent evaluators) policy network
        initializer.

        Args:
            initial_features: Number of input features.
            k_size: Size of first convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            time_window: Size of time window used as agent's state.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.48550/arXiv.1706.10059.
        """
        super().__init__()
        self.device = device

        n_size = time_window - k_size + 1
        self.pred_horizon = prediction_horizon

        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_size),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_size),
            ),
            nn.ReLU(),
        )

        self.final_convolution = nn.Conv2d(
            in_channels=conv_final_features+self.pred_horizon, out_channels=self.pred_horizon, kernel_size=(1, 1)
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=2))

    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation.
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()
        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)
        output = self.sequential(observation)  # shape [BSZ, 20, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [last_stocks, output], dim=1
        )  # shape [BSZ, CONV_FINAL_FEATURES+1, PORTFOLIO_SIZE, 1]
        output = self.final_convolution(output)  # shape [BSZ, T, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # shape [BSZ, T, PORTFOLIO_SIZE+1,1]

        # output shape must be [BSZ, PORTFOLIO_SIZE+1] 
        output = torch.squeeze(output, 3) # shape [BSZ, T, PORTFOLIO_SIZE+1]
        output = self.softmax(output)
        return output

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: Environment observation (dictionary).
          last_action: Last action performed by the agent.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze(0)
        return action

    def _process_last_action(self, last_action):
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
            Last stocks and cash bias.
        """

        batch_size = last_action.shape[0]
        stocks = last_action.shape[2] - 1
        last_stocks = last_action[:, :, 1:].reshape((batch_size, self.pred_horizon, stocks, 1))
        cash_bias = last_action[:, :, 0].reshape((batch_size, self.pred_horizon, 1, 1))
        return last_stocks, cash_bias

