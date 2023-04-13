import numpy as np
import torch
import math
import time

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



def polynomial_example():
  dtype = torch.float
  device = torch.device("cpu")

  x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
  y = torch.sin(x)

  # Create random Tensors for weights. For a third order polynomial, we need
  # 4 weights: y = a + b x + c x^2 + d x^3
  # Setting requires_grad=True indicates that we want to compute gradients with
  # respect to these Tensors during the backward pass.
  #a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
  #b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
  #c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
  #d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

  a = torch.tensor([-0.550234], device=device, dtype=dtype, requires_grad=True)
  b = torch.tensor([-0.960024], device=device, dtype=dtype, requires_grad=True)
  c = torch.tensor([-0.465877], device=device, dtype=dtype, requires_grad=True)
  d = torch.tensor([1.06652], device=device, dtype=dtype, requires_grad=True)

  learning_rate = 1e-6
  loss_hist = []
  start_t = time.perf_counter()
  for t in range(2000):
      # Forward pass: compute predicted y using operations on Tensors.
      y_pred = a + b * x + c * x ** 2 + d * x ** 3

      # Compute and print loss using operations on Tensors.
      # Now loss is a Tensor of shape (1,)
      # loss.item() gets the scalar value held in the loss.
      loss = (y_pred - y).pow(2).sum()
      #if t % 100 == 99:
      #    print(t, '-', loss.item())
      #print("loss:", loss.item())
      loss_hist.append(loss.item())

      # Use autograd to compute the backward pass. This call will compute the
      # gradient of loss with respect to all Tensors with requires_grad=True.
      # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
      # the gradient of the loss with respect to a, b, c, d respectively.
      loss.backward()

      #print("a", a.item(), a.grad)
      #print("b", b.item(), b.grad)
      #print("c", c.item(), c.grad)
      #print("d", d.item(), d.grad)

      with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
      
  end_t = time.perf_counter()
  print(f"time: {(end_t - start_t):.4f}")
  print("final loss: ", loss_hist[-1])


polynomial_example()


def plot_polynomial():
  a = -0.00243357
  b = 0.853247
  c = 0.000370452
  d = -0.0928783

  import matplotlib.pyplot as plt

  x = np.linspace(-math.pi, math.pi, 2000)
  y = np.sin(x)

  f = lambda a, b, c, d, x: a + b*x + c*x**2 + d*x**3

  y_pred = []
  for i in range(len(x)):
    y_pred.append( f(a, b, c, d, x[i]) )

  plt.plot(y_pred)
  plt.plot(y)
  plt.show()

plot_polynomial()