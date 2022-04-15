import torch


class TimestepDropout(torch.nn.Dropout):
    """
    Randomly mask whole -1 dim array
    """

    def forward(self, x: torch.Tensor):
        if not self.training or self.p <= 0:
            return x
        mask = torch.rand(*x.shape[:-1], 1, device=x.device) < self.p
        if self.inplace:
            return x.masked_fill_(mask, 0)
        else:
            return x.masked_fill(mask, 0)
