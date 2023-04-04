#ifndef _Ops_H_
#define _Ops_H_

#include <iostream>
#include <vector>
#include <memory>
#include <set>


using InitShape = std::initializer_list<int>; //allows braced-list initlization - Tenor t = {1,2,3};
using Shape = std::vector<int>;
using Index = std::vector<int>;


class Tensor;

// ------------ Function ---------------------

class Op{
public:
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
    //std::vector<Tensor*> parents;
    Op(Tensor*, Tensor*);
    Op(Tensor*);
    Op();
    virtual Tensor forward();
    //virtual void backward(std::shared_ptr<Tensor> );
    virtual void backward(const Tensor*);
};


class Add_op : public Op{
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
public:
    //std::vector<Tensor*> parents;
    Add_op(Tensor* left, Tensor* right);
    Tensor forward();
    //void backward(std::shared_ptr<Tensor> );
    void backward(const Tensor*);
};


class Sub_op : public Op{
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
public:
    Sub_op(Tensor* left, Tensor* right);
    Tensor forward();
    //void backward(std::shared_ptr<Tensor> );
    void backward(const Tensor*);
};


class Mul_op : public Op{
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
public:
    Mul_op(Tensor* left, Tensor* right);
    Tensor forward();
    //void backward(std::shared_ptr<Tensor>);
    void backward(const Tensor*);
};


class tanh_op : public Op{
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
public:
    tanh_op(Tensor* left);
    Tensor forward();
    void backward(const Tensor*);
};

class exp_op : public Op{
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
public:
    exp_op(Tensor* left);
    Tensor forward();
    void backward(const Tensor*);
};

class dot_op : public Op{
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
public:
    dot_op(Tensor* left, Tensor* right);
    Tensor forward();
    void backward(const Tensor*);
};

#endif