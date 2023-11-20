# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import rs


class Nonlinearity(torch.nn.Module):
    def __init__(self, Rs, act):
        super().__init__()

        self.Rs = rs.simplify(Rs)
        self.Rs_in = Rs
        self.Rs_out = Rs
        self.act = act
        ls = sorted(list(set([x[1] for x in Rs])))
        b = {str(l): torch.nn.Parameter(torch.randn(Rs[0][0])) for l in ls if l != 0}
        self.b = torch.nn.ParameterDict(b)

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''
        *size, n = features.size()
        output = []
        index = 0
        for mul, l in self.Rs_in:
            sub = features.narrow(-1, index, mul * (2 * l + 1)) # [..., u * i]
            index += mul * (2 * l + 1)

            if l == 0:
                sub = self.act(sub) # [..., u * i]
            else:
                sub = sub.reshape(*size, mul, 2 * l + 1) # [..., u, i]
                factor =  self.act(sub.norm(2, dim=-1) + self.b[str(l)]) # [..., u]
                factor = factor.unsqueeze(-1) # [..., u, 1]
                sub = sub * factor # [..., u, i]
                sub = sub.reshape(*size, mul * (2 * l + 1)) # [..., u * i]

            output.append(sub)
        assert index == n

        return torch.cat(output, dim=-1)
