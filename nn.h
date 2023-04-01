


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

    // A modules operator() should call forward()
    virtual Tensor forward(const Tensor* t);

};



class Linear : public Module {

    int in_size;
    int out_size;
    bool use_bias;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

public:
    Linear(int in_size, int out_size, bool use_bias = true);
    void print(std::ostream&, const Linear&) const;

    Tensor forward(const Tensor*);

};


}