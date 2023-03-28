import numpy as np
import torch

x = np.random.randint(5, size=(3,))
x = np.array([11, 12, 13])
print(x.shape)
print(x)
print()

w = np.random.randint(5, size=(3,2))
w = np.array([[1, 2], [3, 4], [5, 6]])
print(w.shape)
print(w)
print()

h = x @ w
print(h.shape)
print(h)

print('-------')

def f(a, b):
  return a @ b

x = torch.tensor(x*1., requires_grad=True)
w = torch.tensor(w*1., requires_grad=True)
#x = torch.randn((), requires_grad=True)
#w = torch.randn((), requires_grad=True)

derivative_fn = f(x,w)
print(derivative_fn)

derivative_fn.backward(torch.Tensor([1, 1]))
print(x.grad)
print(w.grad)


print("---------------")

#x = np.random.randint(5, size=(3, 3, 4))
#w = np.random.randint(5, size=(3, 1, 4))

x = np.random.randint(5, size=(3, 784))
w = np.random.randint(5, size=(784, 16))

out = x @ w #   3 x 3 x 4

print(out.shape)