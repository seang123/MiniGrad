#include <iostream>
#include <vector>
#include <set>


class A{

public:
    std::vector<int> vals;
    A(){
        std::cout << "A()\n";
    };
    virtual void sum(){ std::cout << "base class sum\n"; };
};

class B : public A{
public:
    std::vector<int> vals;
    int a = 0;
    int b = 0;
    B(int a, int b){
        vals.push_back(a);
        vals.push_back(b);
        this->a = a;
        this->b = b;
    }
    void sum(){
        std::cout << a + b << "\n";
    }
};

class C{
public:
    std::vector<int> data;
    A* ctx;
    C(){
        ctx = new B(3, 5);
    }
};


int main()
{

    int i = 0;

    int * y = &i;

    int& z = i;

    std::cout << y << "\n";
    std::cout << *y << "\n";
    std::cout << &z << "\n";

    std::cout << "---\n";

    B b(3, 7);
    B b1(4, 8);
    B b2(12, 13);

    std::vector<B*> bs;
    bs.push_back(&b);
    bs.push_back(&b1);
    bs.push_back(&b2);

    //for(size_t i = bs.size(); i > 0; i--){
    //    B * temp = bs[i];
    for(B* temp : bs){
        temp->sum();
    }
    

    return 0;
}
