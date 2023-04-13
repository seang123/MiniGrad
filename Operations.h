
#ifndef _Operations_H_
#define _Operations_H_

class Tensor;

namespace Ops{

// tanh
Tensor tanh(Tensor& t);

// Exponential 
Tensor exp(Tensor& t);

// Power of 
template <typename T>
Tensor pow(Tensor&, T);
Tensor power(Tensor&, int);

// Square
Tensor square(Tensor& t);

// Sum
//Tensor sum(Tensor& t);

}

#endif