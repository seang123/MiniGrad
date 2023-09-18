

#include <iostream>

#include "Tensor.h"
#include "Operations.h"
#include "nn.h"

/*
Neural network operations 
*/

namespace nn{

Module::Module(int in_size, int out_size, bool use_bias)
    : in_size(in_size)
    , out_size(out_size)
    , use_bias(use_bias){}

Module::Module() = default;

Tensor Module::forward(Tensor&){
    throw std::runtime_error("Module::forward() -- Not implemented");
}

Tensor Module::operator()(Tensor& x){
    return forward(x);
}

std::vector<Tensor*> Module::parameters(){
    return {};
}

Module::~Module() = default;


// ---------------------- Linear -------------------------

/** Initialise a linear/dense/fully-connected layer
 * 
*/
Linear::Linear(int in_size, int out_size, bool use_bias_) 
    : in_size(in_size)
    , out_size(out_size)
    , use_bias(use_bias_)
    , Module(in_size, out_size, use_bias_)
    {
    weight = std::make_shared<Tensor>(Tensor::Uniform(Shape{in_size, out_size}));
    weight->requires_grad(true);
    if( use_bias ){
        //bias = std::make_shared<Tensor> (Shape{out_size});
        bias = std::make_shared<Tensor>(Tensor::Uniform(Shape{out_size}));
        bias->requires_grad(true);
    }
}

Tensor Linear::forward(Tensor& x){
    /*
    Compute the dot product between input x and this->weight
    y = xW^T + b
    Input:  [n, m]
    Weight: [m, p]
    Out:    [n, p]
    */

    Tensor ret = x.dot(*weight);
    if( use_bias ){
        Tensor ret2 = ret + *bias;
        return ret2;
    }
    return ret;
};

Tensor Linear::operator()(Tensor& x){
    return forward(x);
}

std::vector<Tensor*> Linear::parameters(){
    return {weight.get(), bias.get()};
}

}
