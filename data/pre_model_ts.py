import pickle

import torch
import torch.nn as nn


def get_class_weight(labels, output_size=2,):
    """ Compute class weight for class balanced training

    Returns:
        list: list of weights of length OUTPUT_SIZE
    """
    n_labels = torch.zeros(output_size)

    for i in range(output_size):
        n_labels[i] += torch.eq(torch.as_tensor(labels), i).sum()

    weights = n_labels.max() / n_labels

    return weights


def loss_fcn(predictions, labels, training_hparams):
    """
    Computes the loss defined by the dataset
    Args:
        X (torch.tensor): Predictions of the model. Shape (batch, time, n_classes)
        Y (torch.tensor): Targets. Shape (batch, time)
    Returns:
        torch.tensor: loss of each samples. Shape (batch, time)
    """

    # Make the predictions of shape (batch, n_classes, time) such that pytorch will get losses for all time steps
    predictions = predictions.permute(0, 2, 1)

    # Get log probability and compute loss
    log_prob = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss(weight=get_class_weight(labels).to(training_hparams['device']), reduction='none')
    return loss_fn(log_prob(predictions), labels).mean()


class LSTM(nn.Module):
    """ A simple LSTM model

    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.
        input_size (int, optional): The size of the input to the model. Defaults to None. If None, the input size is calculated from the dataset.

    Attributes:
        state_size (int): The size of the hidden state of the LSTM.
        recurrent_layers (int): The number of recurrent layers stacked on each other.
        hidden_depth (int): The number of hidden layers of the classifier MLP (after LSTM).
        hidden_width (int): The width of the hidden layers of the classifier MLP (after LSTM).

    Notes:
        All attributes need to be in the model_hparams dictionary.
    """

    def __init__(self, model_hparams, input_size=1, output_size=2):
        super(LSTM, self).__init__()

        ## Save stuff
        # Model parameters
        self.device = model_hparams['device']
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']

        # Dataset parameters
        self.input_size = input_size
        self.output_size = output_size

        ## Recurrent model
        self.lstm = nn.LSTM(self.input_size, self.state_size, self.recurrent_layers, batch_first=True)

        ## Classification model
        layers = []
        if self.hidden_depth == 0:
            layers.append(nn.Linear(self.state_size, self.output_size))
        else:
            layers.append(nn.Linear(self.state_size, self.hidden_width))
            for i in range(self.hidden_depth - 1):
                layers.append(nn.Linear(self.hidden_width, self.hidden_width))
            layers.append(nn.Linear(self.hidden_width, self.output_size))

        seq_arr = []
        for i, lin in enumerate(layers):
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input):
        """ Forward pass of the model

        Args:
            input (torch.Tensor): The input to the model.

        Returns:
            torch.Tensor: The output of the model.
        """

        # Get prediction steps
        pred_time = 49
        # Setup hidden state
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward propagate LSTM
        input = input.view(input.shape[0], input.shape[1], -1)
        features, hidden = self.lstm(input, hidden)

        # Extract features at prediction times
        all_features = torch.zeros((input.shape[0], pred_time.shape[0], features.shape[-1])).to(input.device)
        for i, t in enumerate(pred_time):
            all_features[:, i, ...] = features[:, t, ...]

        # Make prediction with fully connected
        all_out = self.classify(all_features)

        return all_out, all_features

    def classify(self, features):

        n_pred = features.shape[1]
        all_out = torch.zeros((features.shape[0], n_pred, self.output_size)).to(features.device)

        for t in range(n_pred):
            output = self.classifier(features[:, t, :])
            all_out[:, t, ...] = output

        return all_out

    def initHidden(self, batch_size, device):
        """ Initialize the hidden state of the LSTM with a normal distribution

        Args:
            batch_size (int): The batch size of the model.
            device (torch.device): The device to use.
        """
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device),
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

    def get_classifier_network(self):
        return self.classifier


if __name__ == "__main__":

    model_params = {'state_size': 20,
                    "hidden_width": 20,
                    'hidden_depth': 2,
                    "recurrent_layers": 1,
                    'class_balance': True,
                    'weight_decay': 0,
                    'lr': 1e-3,
                    'batch_size': 64,
                    "device": "cpu"}
    model = LSTM(model_params)