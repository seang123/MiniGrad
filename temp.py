import numpy as np
import torch

np.random.seed(42)

def func1():
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

def func2():
  #x = np.random.randint(5, size=(3, 3, 4))
  #w = np.random.randint(5, size=(3, 1, 4))

  x = np.random.randint(5, size=(3, 784))
  w = np.random.randint(5, size=(784, 16))

  out = x @ w #   3 x 3 x 4

  print(out.shape)

  print("---------------------------")

def kaparthy_example():
  a = torch.tensor(1.0 * np.random.randint(5, size=(5,)), requires_grad=True)
  b = torch.tensor(1.0 * np.random.randint(5, size=(5,)), requires_grad=True)
  c = torch.tensor(1.0 *np.random.randint(5, size=(5,)), requires_grad=True)
  e = a * b
  e.retain_grad()
  d = e + c
  d.retain_grad()
  f = -1.0 * np.array([2, 2, 2, 2, 2])
  f = torch.tensor(f, requires_grad=True)
  L = d * f

  L.backward(torch.Tensor([1, 1, 1, 1, 1]))

  print("a: ", a)
  print("b: ", b)
  print("c: ", c)
  print("e: ", e)
  print("d: ", d)
  print("L: ", L)

  print("-- grad -- ")
  print(a.grad)
  print(b.grad)
  print(c.grad)
  print(e.grad)
  print(d.grad)
  print(f.grad)

def tanh_exp():
  #t = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5])
  #t = np.tanh(t)
  #print(t)
  #print("----")

  t = torch.tensor(np.array([-1., 0., 1., 2., 3., 4., 5.]), requires_grad=True)

  #out = torch.tanh(t);
  out = torch.exp(t)
  print(out)
  out.backward(torch.Tensor([1, 1, 1, 1, 1, 1, 1]));

  print(t.grad)




def linear_layer():

  #x = np.random.uniform(0, 1, size=(16, 32))
  #x.astype(np.float32)
  #x = torch.tensor(x)

  x = torch.tensor(np.array([[1., 2.], [3., 4.]]), requires_grad=True)
  y = torch.tensor(np.array([[1., 2., 3.], [4., 5., 6.]]), requires_grad=True)

  out = x @ y;
  print(out)

  out.backward(torch.tensor([[1, 1, 1], [1, 1, 1]]))
  print(x.grad)
  print(y.grad)





linear_layer()


