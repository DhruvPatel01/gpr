# @Author: sachan
# @Date:   2019-03-09T22:29:01+05:30
# @Last modified by:   sachan
# @Last modified time: 2019-03-10T00:17:45+05:30

import torch
import torch.nn.functional as F


class gap_model1(torch.nn.Module):
    """ Bilinear model with softmax"""

    def __init__(self, embedding_size):
        super(gap_model1, self).__init__()
        self.embedding_size = embedding_size
        self.W = torch.randn((embedding_size, embedding_size), requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    def forward(self, x1, x2):
        bilinear_score = torch.mm(x1, torch.mm(self.W, x2.t())) + self.b
        softmax_score = F.softmax(bilinear_score, dim=0)
        return softmax_score

if __name__ == '__main__':
    x1 = torch.randn(1000,764)
    x2 = torch.randn(1,764)

    model = gap_model1(764)
    pred = model.forward(x1, x2)
    print(torch.argmax(pred))
    print(sum(pred))
    print(pred.shape)
