# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''

        return features / features.norm(2, dim=-1, keepdim=True)  # [..., u*i]
