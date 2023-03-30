
#ifndef _Operations_H_
#define _Operations_H_

class Tensor;

namespace Ops{

// tanh
Tensor tanh(Tensor& t);

// Exponential 
Tensor exp(Tensor& t);

// Power of 
Tensor pow(Tensor& t);

// Square
Tensor square(Tensor& t);

}

#endif