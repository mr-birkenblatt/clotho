from torch.distributions.distribution import Distribution as Distribution


class ExponentialFamily(Distribution):
    def entropy(self): ...
