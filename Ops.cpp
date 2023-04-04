
#include <iostream>
#include <vector>
#include <memory>

#include "Ops.h"
#include "Tensor.h"

class Tensor;

Op::Op() = default;

Op::Op(Tensor* left){
    this->left = left;
    parents.insert(left);
}

Op::Op(Tensor* left, Tensor* right){
    this->left = left;
    this->right = right;
    parents.insert(left);
    parents.insert(right);
}

void Op::backward(const Tensor* out){
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
void Add_op::backward(const Tensor* out){
    if(left->requires_grad()){
        // compute gradient for left parent
        //* left.grad += 1. * out.grad
        left->grad = out->grad;
        //* left.grad.shape == left.shape   (may require reshape)
        //left->grad = std::make_shared<Tensor>(left->grad->reshape(left->shape()));
    }
    if(right->requires_grad()){
        // compute gradient for right parent
        //* right.grad += 1. * out.grad
        right->grad = out->grad;
        //* right.grad.shape == right.shape   (may require reshape)
        //right->grad = std::make_shared<Tensor>(right->grad->reshape(right->shape()));
    }
}

Tensor Add_op::forward(){
    throw std::runtime_error("Add_op::forward() -- Not implemented!");
}

// ----------------- Subtraction ----------------------

Sub_op::Sub_op(Tensor* left, Tensor* right) : Op(left, right){
    this->left = left;
    this->right = right;
    parents.insert(left);
    parents.insert(right);
}

void Sub_op::backward(const Tensor* out){
    if(left->requires_grad()){
        // compute gradient for left parent
        //* left.grad += 1. * out.grad
        left->grad = out->grad;
        //* left.grad.shape == left.shape   (may require reshape)
        //left->grad = std::make_shared<Tensor>(left->grad->reshape(left->shape()));
    }
    if(right->requires_grad()){
        // compute gradient for right parent
        //* right.grad += 1. * out.grad
        right->grad = out->grad;
        //* right.grad.shape == right.shape   (may require reshape)
        //right->grad = std::make_shared<Tensor>(right->grad->reshape(right->shape()));
    }
}

Tensor Sub_op::forward(){
    throw std::runtime_error("Sub_op::forward() -- Not implemented!");
}

// ----------------- Multiplication ----------------------

Mul_op::Mul_op(Tensor* left, Tensor* right) : Op(left, right){
    this->left = left;
    this->right = right;
    parents.insert(left);
    parents.insert(right);
}

void Mul_op::backward(const Tensor* out){
    if(left->requires_grad()){
        //* left.grad = right.values * out_grad
        left->grad = std::make_shared<Tensor>((*right) * (*out->grad));
    }
    if(right->requires_grad()){
        //* right.grad = left.values * out_grad
        right->grad = std::make_shared<Tensor>((*left) * (*out->grad));
    }
}

Tensor Mul_op::forward(){
    throw std::runtime_error("Mul_op::forward() -- Not implemented!");
}


// ----------------- tanh ----------------------

tanh_op::tanh_op(Tensor* left) : Op(left){
    this->left = left;
    parents.insert(left);
}

void tanh_op::backward(const Tensor* out){
    if(left->requires_grad()){
        //* left.grad += (1 - t**2) * out.grad
        Tensor squared = out->square();
        Tensor temp = 1.f - squared;
        left->grad = std::make_shared<Tensor>( temp * (*out->grad) );
    }
}

Tensor tanh_op::forward(){
    throw std::runtime_error("tanh_op::forward() -- Not implemented!");
}

// ---------------- exp e^ ------------------


exp_op::exp_op(Tensor* left) : Op(left){
    this->left = left;
    this->parents.insert(left);
}

void exp_op::backward(const Tensor* out){
    if(left->requires_grad()){
        Tensor copy = *out;
        left->grad = std::make_shared<Tensor>(copy);
    }
}

Tensor exp_op::forward(){
    throw std::runtime_error("exp_op::forward() -- Not implemented!");
}

// ---------------- dot . product ------------------


dot_op::dot_op(Tensor* left, Tensor* right) : Op(left, right){
    this->left = left;
    this->right = right;
    this->parents.insert(left);
    this->parents.insert(right);
}

/*
    Y = (X^T)W + B
    dL/dW = (dL/dW ^ T) X
    dL/dX = (dL/dW) W

    So 
    dL/dW = out->grad * left
    dL/dX = out->grad * right

    out->grad is the upstream gradient which will be computed first in the operations chain

    in:  [32, 3, 16] @ [16, 2]
    out: [32, 3, 2]
    out->grad: [32, 3, 2]  // (1)
    right:     [16, 2]    //  (2) cannot multiply 1 and 2


    in:  [2, 2] @ [2, 3]
    out: [2, 3]
    out->grad: [2, 3]
*/
void dot_op::backward(const Tensor* out){
    if(left->requires_grad()){
        //left->grad = std::make_shared<Tensor>(*(out->grad) * *right);
        // [2, 3] * [3, 2] => [2, 2]
        //left->grad = std::make_shared<Tensor>(out->grad->dot(*right));
        Tensor right_t = Transpose(*right);
        left->grad = std::make_shared<Tensor>(out->grad->dot(right_t));
    }
    if(right->requires_grad()){
        //right->grad = std::make_shared<Tensor>(*left * *(out->grad));
        // [2, 2] * [2, 3]
        //right->grad = std::make_shared<Tensor>(left->dot(*out->grad));
        Tensor left_t = Transpose(*left);
        right->grad = std::make_shared<Tensor>(left_t.dot(*out->grad));
    }
}

Tensor dot_op::forward(){
    throw std::runtime_error("dot_op::forward() -- Not implemented!");
}