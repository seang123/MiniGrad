#include "Tensor.h"

#include <vector>
#include <algorithm>
#include <math.h>


template <typename F>
static void runOp(int size, F op){
    for(int i = 0; i < size; i++){
        op(i);
    }
}

template <typename F>
inline void ApplyOpSimple(Tensor& ret, const Tensor& src, F op) {
    auto&& ret_data = ret.data();
    auto&& src_data = src.data();
    // Simply apply all
    for(size_t i = 0; i < src.size(); i++){
        ret_data[i] = op(src_data[i]);
    }
}

namespace Ops{


template <typename T>
static float _tanh(T y){
    return (exp(2 * y) - 1 ) / (exp(2 * y) + 1);
}

template <typename T>
static float _exp(T y){
    return exp(y);
}

/**
 * Computes tanh on a given tensor and returns a new tensor
*/
Tensor tanh(Tensor& t){
    Tensor ret (t.shape());
    ApplyOpSimple(ret, t, _tanh<float>);

    if(t.requires_grad()){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<tanh_op>(&t);
    }

    return ret;
}

/**
 * Computes exp (e^) on a given tensor and returns a new tensor
*/
Tensor exp(Tensor& t){
    Tensor ret (t.shape());
    ApplyOpSimple(ret, t, _exp<float>);

    if(t.requires_grad()){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<exp_op>(&t);
    }

    return ret;
}

/**
 * A functor for computing the power of
*/
template <typename T>
class _pow{
    T p = 0;
public:
    _pow(T p){
        this->p = p;
    }
    float operator()(float b){
        return std::pow(b, p);
    }
};

/**
 * Takes the tensor to the power of some value
*/
template <typename T>
Tensor pow(Tensor& t, T p){
    Tensor ret (t.shape());
    _pow<T> to_the_power(p);
    ApplyOpSimple(ret, t, to_the_power);
    return ret;
}

/**
 * Square the values in a tensor
*/
Tensor square(Tensor& t){
    Tensor ret (t.shape());
    _pow<float> to_the_power(2);
    ApplyOpSimple(ret, t, to_the_power);
    return ret;
}

}