import pickle
import torch
import argparse

from pre_model_ts import loss_fcn, LSTM


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./')
    opt = parser.parse_args()

    return opt


def train(model, device, train_loader, optimizer, epoch):

    model.train()
    model.to(device)

    for _ in epoch:
        for idx, (data, target) in enumerate(train_loader):
            pred = model(data.to(device))
            loss = loss_fcn(pred, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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



