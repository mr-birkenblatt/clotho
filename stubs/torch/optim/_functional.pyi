from typing import List

from torch import Tensor as Tensor

from .adadelta import adadelta as adadelta
from .adagrad import adagrad as adagrad
from .adam import adam as adam
from .adamax import adamax as adamax
from .adamw import adamw as adamw
from .asgd import asgd as asgd
from .nadam import nadam as nadam
from .radam import radam as radam
from .rmsprop import rmsprop as rmsprop
from .rprop import rprop as rprop
from .sgd import sgd as sgd


def sparse_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], state_steps: List[int], *, eps: float, beta1: float, beta2: float, lr: float): ...