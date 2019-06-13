# coding:utf8
import torch


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(self)[:-2]

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
