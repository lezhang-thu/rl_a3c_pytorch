from __future__ import division
from model import A3Clstm
import numpy as np
import torch
import json
import logging
import torch.nn as nn


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    init_bias = 0.0
    if type(m) == nn.Conv2d:
        gain = nn.init.calculate_gain('relu')
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, init_bias)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=gain)
    elif type(m) == nn.LSTMCell:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, init_bias)
    elif type(m) == A3Clstm:
        for name, param in m.named_parameters():
            if 'critic_linear' in name or 'actor_linear' in name:
                if 'weight' in name:
                    std = 1.0 if 'critic_linear' in name else 0.01
                    nn.init.normal_(param)
                    with torch.no_grad():
                        param.mul_(
                            std / torch.norm(param, dim=1, keepdim=True))
                elif 'bias' in name:
                    nn.init.constant_(param, init_bias)
