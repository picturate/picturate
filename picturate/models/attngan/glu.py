from picturate.imports import *
from picturate.models.attngan import *


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels dont divide 2!"
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])
