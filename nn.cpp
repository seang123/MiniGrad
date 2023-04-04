

#include "Tensor.h"
#include "Operations.h"
#include "nn.h"

/*
Neural network operations 
*/

namespace nn{


// ---------------------- Linear -------------------------

/** Initialise a linear/dense/fully-connected layer
 * 
*/
Linear::Linear(int in_size, int out_size, bool use_bias) : Module() {
    this->in_size = in_size;
    this->out_size = out_size;
    this->use_bias = use_bias;
    weight = std::make_shared<Tensor> (Shape(out_size, in_size));
    if( use_bias ){
        bias = std::make_shared<Tensor> (Shape(out_size));
    }
}

Tensor Linear::forward(const Tensor& x){
    // Compute the dot product between input x and this->weight
    // y = xW^T + b

    /*
    Input:  [n, m]
    Weight: [m, p]
    Out:    [n, p]
    */
    Tensor ret = x.dot(*weight);
    if( use_bias ){
        ret = ret + *bias;
    }
    return ret;
};

/**
 * Print a Linear layers in/out shape
*/
void Linear::print(std::ostream& os, const Linear& l) const{
    os << "(" << this->in_size << ", " << this->out_size << ")";
}

/**
 * Overloaded print operator
*/
std::ostream& operator<<(std::ostream& os, const Linear& l) {
    l.print(os, l);
    return os;
}

}