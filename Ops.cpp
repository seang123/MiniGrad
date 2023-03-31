
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
    if(left->requires_grad()){
        // compute gradient for left parent
        //* left.grad += 1. * out.grad
        left->grad = out_grad;
        //* left.grad.shape == left.shape   (may require reshape)
        //left->grad = std::make_shared<Tensor>(left->grad->reshape(left->shape()));
    }
    if(right->requires_grad()){
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

// ----------------- Subtraction ----------------------

Sub_op::Sub_op(Tensor* left, Tensor* right) : Op(left, right){
    this->left = left;
    this->right = right;
    parents.insert(left);
    parents.insert(right);
}

void Sub_op::backward(std::shared_ptr<Tensor> out_grad){
    if(left->requires_grad()){
        // compute gradient for left parent
        //* left.grad += 1. * out.grad
        left->grad = out_grad;
        //* left.grad.shape == left.shape   (may require reshape)
        //left->grad = std::make_shared<Tensor>(left->grad->reshape(left->shape()));
    }
    if(right->requires_grad()){
        // compute gradient for right parent
        //* right.grad += 1. * out.grad
        right->grad = out_grad;
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


// ----------------- tanh ----------------------

tanh_op::tanh_op(Tensor* left) : Op(left){
    this->left = left;
    parents.insert(left);
}

void tanh_op::backward(std::shared_ptr<Tensor> out_grad){
    if(left->requires_grad()){
        //* left.grad += (1 - t**2) * out.grad
        Tensor squared = left->square();  //! TODO: We should square the value of the child not the parent
        Tensor temp = 1.f - squared;
        //std::cout << "left: " << *left << "\n";
        //std::cout << "squared: " << squared << "\n";
        std::cout << "temp: " << temp << "\n";
        left->grad = std::make_shared<Tensor>( temp * (*out_grad) );
    }
}

Tensor tanh_op::forward(){
    throw std::runtime_error("tanh_op::forward() -- Not implemented!");
}