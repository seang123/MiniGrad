#include <iostream>
#include <vector>
#include <set>
#include <memory>


class Tensor;
class Op;

class Tensor{
public:
    std::shared_ptr<Op> op;
    int x;
    Tensor(int y): x(y) {}
    Tensor(Tensor&& lhs)
    : x(std::move(lhs.x))
    , op(std::move(lhs.op))
    {
        std::cout << "X after move: " << x << "\n";
    }
    Tensor& operator=(const Tensor& lhs) { 
        std::cout << "Tensor& operator=(const Tensor&)\n"; 
        return *this; 
    };
    Tensor& operator=(Tensor&& lhs){
        std::cout << "Tensor::operator=\n"; 
        return *this; 
    }
    Tensor(const Tensor& lhs) : op(lhs.op) , x(lhs.x) { std::cout << "Tensor(const Tensor& lhs)\n"; }
};

std::ostream& operator<<(std::ostream& os, const Tensor& x) {
    std::cout << x;
    return os;
}

std::ostream& operator<<(std::ostream& os, std::shared_ptr<Tensor> t) {
    std::cout << t->x;
    return os;
}

class Op{
public:
    //Tensor * lhs;
    //Tensor * rhs;
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::vector<Tensor> parents;
    //Op(Tensor * lhs, Tensor * rhs) 
    Op(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs) 
        : lhs(lhs)
        , rhs(rhs)
        {std::cout << "Op(Tensor&&, Tensor&&)\n";}

    Op(Op&& o) = default;
};

Tensor operator*(Tensor&& lhs, Tensor&& rhs){
    std::shared_ptr<Tensor> lhs_ = std::make_shared<Tensor>(lhs);
    std::shared_ptr<Tensor> rhs_ = std::make_shared<Tensor>(rhs);
    Tensor ret(11);
    // ... do addition here ...
    //ret.op = std::make_shared<Op>(rhs_);
    ret.op = std::make_shared<Op>(lhs_, rhs_);
    return ret;
}



int main()
{
    std::cout << "---\n";


    // Tensor a = f(Tensor(4));
    Tensor a = Tensor(3) * Tensor(4); 
    
    std::cout << "--- out ---\n";
    std::cout << a.x << "\n";
    std::cout << a.op->lhs << "\n";;
    std::cout << a.op->rhs << "\n";;

    std::cout << "---\n";
    return 0;
}
