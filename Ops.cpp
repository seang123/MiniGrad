
#include <iostream>
#include <vector>
#include <memory>

#include "Ops.h"
#include "Tensor.h"

class Tensor;

Op::Op() = default;

Op::Op(Tensor* left, Tensor* right){
    this->left = left;
    this->right = right;
    parents.insert(left);
    parents.insert(right);
}

void Op::backward(){
    //throw std::runtime_error("Op.backward() not implemented!");
    std::cout << "Op::backward()\n";
}

Tensor Op::forward(){
    throw std::runtime_error("Op.forward() not implemented!");
}


// ------------------- Addition operator ---------------------

Add_op::Add_op(Tensor* left, Tensor* right) : Op(left, right){
    this->left = left;
    this->right = right;
    //parents.push_back(left);
    //parents.push_back(right);
    parents.insert(left);
    parents.insert(right);
    std::cout << "add_op parents: "<< parents.size() << "\n";
}

/**
 * Compute the gradients for the parents of a tensor
*/
void Add_op::backward(){
    std::cout << "Add_op::backward()\n";
    if(left->requires_grad()){
        // compute gradient for left parent
        //* left.grad = 1. * out.grad
        
        //* left.grad.shape == left.shape   (may require reshape)
    }
    if(right->requires_grad()){
        // compute gradient for right parent
        //* right.grad = 1. * out.grad

        //* right.grad.shape == right.shape   (may require reshape)
    }
}

Tensor Add_op::forward(){
    throw std::runtime_error("Not implemented!");
}