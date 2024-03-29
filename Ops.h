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
    Tensor* self;
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
    //std::vector<Tensor*> parents;
    Op(Tensor*, Tensor*);
    Op(Tensor*);
    Op(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>);
    Op();
    virtual Tensor forward();
    //virtual void backward(std::shared_ptr<Tensor> );
    virtual void backward(const Tensor*);
};


class Add_op : public Op{
    Tensor * self;
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
public:
    //std::vector<Tensor*> parents;
    Add_op(Tensor* self, Tensor* left, Tensor* right);
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
    void backward(const Tensor*);
};



class div_op : public Op{
    Tensor* left;
    Tensor* right;
    std::set<Tensor*> parents;
public:
    div_op(Tensor* left, Tensor* right);
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

class pow_op : public Op{
    Tensor* left;
    Tensor* right;
    float pow;
    std::set<Tensor*> parents;
public:
    pow_op(Tensor* left, float pow);
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


// ---------------- Temporary op's -------------------

class Add_op_t : public Op{
    std::shared_ptr<Tensor> left;
    std::shared_ptr<Tensor> right;
    //std::set<Tensor*> parents;
    std::set<std::shared_ptr<Tensor>> parents;
public:
    Add_op_t(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>);
    Tensor forward();
    void backward(const Tensor*);
};

class Mul_op_t : public Op{
    std::shared_ptr<Tensor> left;
    std::shared_ptr<Tensor> right;
    //std::set<Tensor*> parents;
    std::set<std::shared_ptr<Tensor>> parents;
public:
    Mul_op_t(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>);
    Tensor forward();
    void backward(const Tensor*);
};

#endif