
const after function definition - makes it a compiler error for this class function to change a member variable of the class
reading is allowed.


Based on work from: https://github.com/takiyu/tinyndarray


Run:

    g++ -std=c++17 -static main.cpp Tensor.cpp Substance.cpp Iter.cpp



# Operations

Base function which takes an operation (+-*/) and two tensors as parameters.
Returns a new tensor (reference?).

# Gradients

Separate class like substance which stores the gradients

# Batching

no idea

# Issues

Currently cannot create an array with a final shape dimension of 1 ie. [2, 3, 1]
work around is create [2, 3] and then expand dimension