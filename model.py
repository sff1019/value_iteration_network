"""
Referenced: https://github.com/zuoxingdong/VIN_PyTorch_Visdom/blob/master
"""
import torch
import torch.nn as nn


def attention(tensor, S1, S2, img_size, ch_q):
    n_data = tensor.shape[0]

    # sliceing s1 position
    slice_s1 = S1.expand(img_size, 1, ch_q, n_data)
    slice_s1 = slice_s1.permute(3, 2, 1, 0)
    q_out = tensor.gather(2, slice_s1).squeeze(2)

    # slicing s2 position
    slice_s2 = S2.expand(1, ch_q, n_data)
    slice_s2 = slice_s2.permute(2, 1, 0)
    q_out = q_out.gather(2, slice_s2).squeeze(2)

    return q_out


class VIN(nn.Module):
    def __init__(self, ch_in=2, ch_h=150, ch_q=10):
        super(VIN, self).__init__()
        # number of channels in q layer (~action)
        self.ch_q = ch_q

        # first hidden conv layer
        self.h = nn.Conv2d(2, ch_h, kernel_size=3, padding=1)
        # conv layer to generate reward image
        self.r = nn.Conv2d(ch_h, 1, kernel_size=3, padding=1)
        # q function in VI module
        self.q = nn.Conv2d(2, ch_q, kernel_size=3, padding=1)

        self.fc = nn.Linear(ch_q, 8)

    def forward(self, x, S1, S2, k=10):
        # get reward image from observation image
        h = self.h(x)
        r = self.r(h)

        v = torch.zeros(r.size()).to(x.device)

        for _ in range(k):
            rv = torch.cat([r, v], dim=1)  # [batch_size, 2, img_size, img_size]
            q = self.q(rv)
            v, _ = torch.max(q, dim=1)
            v = v.unsqueeze(1)

        rv = torch.cat([r, v], dim=1)
        q = self.q(rv)

        # attention model
        out = attention(q, S1, S2, q.shape[2], self.ch_q)

        # fully connected layer
        out = self.fc(out)

        return out
