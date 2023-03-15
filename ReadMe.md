
# Tensor

A simple C++ ndarray autograd library

--- 

The ndarray implementation is based on work from: https://github.com/takiyu/tinyndarray

---

Compile:

    g++ -std=c++17 -static main.cpp Tensor.cpp Substance.cpp Iter.cpp Ops.cpp



const after function definition - makes it a compiler error for this class function to change a member variable of the class
reading is allowed.


# Operations

Base function which takes an operation (+-*/) and two tensors as parameters.
Returns a new tensor (reference?).

# Gradients

Separate class like substance which stores the gradients

# Batching

no idea

# Issues

> Currently cannot create an array with a final shape dimension of 1 ie. [2, 3, 1]
work around is create [2, 3] and then expand dimension

> Calling .requires_grad(true) on a tensor created in the ApplyDualOp() method changes the requires_grad attribute to true, however this doesn't hold for the returned tensor for some reason.