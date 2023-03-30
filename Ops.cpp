
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

void Op::backward(std::shared_ptr<Tensor> out_grad){
    throw std::runtime_error("Op.backward() not implemented!");
    //std::cout << "Op::backward()\n";
}

Tensor Op::forward(){
    throw std::runtime_error("Op.forward() not implemented!");
}


// ------------------- Addition operator ---------------------

Add_op::Add_op(Tensor* left, Tensor* right) : Op(left, right){
    this->left = left;
    this->right = right;
    parents.insert(left);
    parents.insert(right);
}

/**
 * Compute the gradients for the parents of a tensor
*/
void Add_op::backward(std::shared_ptr<Tensor> out_grad){
    std::cout << "Add_op::backward()\n";
    if(left->requires_grad()){
        std::cout << "left\n";
        // compute gradient for left parent
        //* left.grad += 1. * out.grad
        left->grad = out_grad;
        //* left.grad.shape == left.shape   (may require reshape)
        //left->grad = std::make_shared<Tensor>(left->grad->reshape(left->shape()));
    }
    if(right->requires_grad()){
        std::cout << "right\n";
        // compute gradient for right parent
        //* right.grad += 1. * out.grad
        right->grad = out_grad;
        //* right.grad.shape == right.shape   (may require reshape)
        //right->grad = std::make_shared<Tensor>(right->grad->reshape(right->shape()));
    }
}

Tensor Add_op::forward(){
    throw std::runtime_error("Add_op::forward() -- Not implemented!");
}


// ----------------- Multiplication ----------------------

Mul_op::Mul_op(Tensor* left, Tensor* right) : Op(left, right){
    this->left = left;
    this->right = right;
    parents.insert(left);
    parents.insert(right);
}

void Mul_op::backward(std::shared_ptr<Tensor> out_grad){
    if(left->requires_grad()){
        //* left.grad = right.values * out_grad
        left->grad = std::make_shared<Tensor>((*right) * (*out_grad));
    }
    if(right->requires_grad()){
        //* right.grad = left.values * out_grad
        right->grad = std::make_shared<Tensor>((*left) * (*out_grad));
    }
}

Tensor Mul_op::forward(){
    throw std::runtime_error("Mul_op::forward() -- Not implemented!");
}