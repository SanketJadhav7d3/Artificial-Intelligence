
from lib.tensor import Tensor
from lib.deriv.derivative import deriv
import numpy as np

a = Tensor([1, 2, 3, 4, 5], autograd=True)
b = Tensor([1, 2, 3, 4, 5], autograd=True)
c = Tensor([1, 2, 3, 4, 5], autograd=True)

print(c.sum(0))
