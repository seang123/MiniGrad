#ifndef _Ops_H_
#define _Ops_H_

#include <iostream>
#include <vector>
#include <memory>


using InitShape = std::initializer_list<int>; //allows braced-list initlization - Tenor t = {1,2,3};
using Shape = std::vector<int>;
using Index = std::vector<int>;


class Tensor;

// ------------ Function ---------------------

class Op{
public:
    virtual Tensor forward();
    virtual void backward();
};

class Add : public Op{
private:
    Tensor* self_;
    Tensor* other_;
public:
    Add(Tensor* self, Tensor* other);
    void backward();
    Tensor forward();
};

class Sub : public Op{
private:
    const Tensor& lhs;
    const Tensor& rhs;
public:
    Sub(const Tensor& lhs, const Tensor& rhs);
    void backward();
};

#endif