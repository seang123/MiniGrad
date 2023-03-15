
#include <iostream>
#include <vector>
#include <memory>

#include "Ops.h"
#include "Tensor.h"

class Tensor;



void Op::backward(){
    throw std::runtime_error("Not implemented!");
}

Tensor Op::forward(){
    throw std::runtime_error("Not implemented!");
}


// ------------------- Addition operator ---------------------

Add::Add(Tensor* self, Tensor* other){
    self_ = self;
    other_ = other;
}

void Add::backward(){
    throw std::runtime_error("Not implemented!");
}

Tensor Add::forward(){
    throw std::runtime_error("Not implemented!");
}