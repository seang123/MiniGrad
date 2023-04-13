
# Tensor

A simple C++ ndarray autograd library

--- 

The ndarray implementation is based on work from: https://github.com/takiyu/tinyndarray

---

Compile with makefile or:

    g++ -std=c++17 -static main.cpp Tensor.cpp Substance.cpp Ops.cpp Operations.cpp


---

Example use: 

```
Tensor t1 = {1.f, 2.f, 3.f, 4.f, 5.f}; // shape: (5,)
Tensor t2 = {5.f}; // shape: (1,)

// Forward pass 
Tensor out = t1 * t2;

// Backwards pass
out.backward();

// Get the gradients (stored as a tensor obj)
std::cout << *t1.grad << "\n";

```

--- 

# Initialisation

```
// Tensor with some shape (buffer will have random undefined values)
Tensor t = Tensor(Shape{2, 3}); // shape: (2, 3) 

// Tensor of 1's
Tensor t = Tensor::Ones(Shape(2, 3)) 
Tensor t = Tensor::Ones(3, 2, 4) // shape: (3, 2, 4)

Tensor::Seed(42); // Seed rng 
Tensor t = Tensor::Uniform(0.f, 1.f, Shape(2, 3)); // Uniform [0., 1.] shape: (2,3)
Tensor t = Tensor::Uniform((2, 3)); // Uniform [0., 1.] shape: (2, 3)
```

# Operations

Base function which takes an operation (+-*/) and two tensors as parameters.
Returns a new tensor.


# Gradients

The output tensor holds references to its parents and what operation created(class Op) it

There is a method Tensor::backward() which sorts the graph and then applies the gradient updates by calling the ctx.backwards() method

The Op class (ctx) has a backward() method which updates the weights of the parents

# Batching

TODO

# Issues

- [ ] Currently cannot create an array with a final shape dimension of 1 ie. 2x3x1
work around is to create a 2x3 and then reshape to 2x3x1

- [ ] Calling .requires_grad(true) on a tensor created in the ApplyDualOp() method changes the requires_grad attribute to true, however this doesn't hold for the returned tensor for some reason.

    - Think this was becuase of the move constructor only copying the substance and nothing else / think I fixed it but haven't checked


# TODO

- [ ] neural network modules (linear, conv, ...)

- [ ] Add support for rvalue-references in the tensor operators

- [ ] Test if .backward() works properly if a tensor is the parent for > 1 operation

- [ ] Static graph bulding

- [ ] refactor some of the operations methods to take const parameters

# Notes

Surprisingly converting the tanh operation to use SIMD instructions while still only working on 1 value at a time is much faster than what the compiler can 
achieve with -O3. However, working one 4 values at a time didn't seem to give any speeds ups, so the compiler must be unrolling the loop already.



# R-value refs and gradients

Tensor operator+(Tensor&& lhs, Tensor&& rhs) - allowes for following notation:
Tensor a; Tensor b; Tensor c;
Tensor d = (a + b) + c;
(a + b) results in a temporary value (Tensor&&) which is added to a, in this case, l-value tensor 'c'.
(a + b) result has a ctx and will compute gradients