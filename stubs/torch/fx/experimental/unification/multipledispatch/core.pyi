from _typeshed import Incomplete

from .dispatcher import Dispatcher as Dispatcher
from .dispatcher import MethodDispatcher as MethodDispatcher


global_namespace: Incomplete

def dispatch(*types, **kwargs): ...
def ismethod(func): ...
