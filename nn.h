
#ifndef _nn_H_
#define _nn_H_



#include "Tensor.h"
#include "Operations.h"

/*
Neural network operations 
*/

namespace nn{

/** Base class implemented by all neural network modules
 * 
*/
class Module{

    int in_size;
    int out_size;
    int use_bias;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

public:
    Module();
    Module(int, int, bool);

    // A modules operator() should call forward()
    virtual Tensor forward(Tensor&);
    virtual Tensor operator()(Tensor&);

};



class Linear : public Module {

    int in_size;
    int out_size;
    bool use_bias;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

public:
    Linear(int in_size, int out_size, bool use_bias_ = true);

    Tensor forward(Tensor&);
    Tensor operator()(Tensor&);

    std::shared_ptr<Tensor> get_weights();
};


}


#endif