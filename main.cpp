#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <immintrin.h>

#include "Tensor.h"
#include "Operations.h"
#include "nn.h"

//#define TINYNDARRAY_IMPLEMENTATION
//#include "tinyndarray.h"
//using tinyndarray::NdArray;

template <typename DT = std::chrono::microseconds,
          typename ClockT = std::chrono::steady_clock>
class Timer{
    using timep_t = decltype(ClockT::now());
    timep_t _start = ClockT::now();
    timep_t _end   = {};
public:
    void tic(){
        _end = timep_t{};
        _start = ClockT::now();
    }
    void toc(){
        _end = ClockT::now();
    }

    template <typename duration_t = DT>
    auto duration() const {
        //    std::cout << clock.duration().count() << "\n";
        assert( _end != timep_t{} && "Timer must be call .toc() before reading time");
        return std::chrono::duration_cast<duration_t>(_end - _start);
    }
};

using std::cout;

void basic_test(){

    //Tensor t4(std::vector<float> {1, 2, 3, 8});
    Tensor t4 = {
        {11, 12, 13}, 
        {4, 5, 6}
    };

    /*Tensor t1 = {
        {1, 2, 3},
        {6, 5, 4}
    };*/

    Tensor t1 = {2.0f, 1.0f};
    t1 = t1.reshape(2, 1);
    //Tensor t1 = {{2.0f, 3.f, 4.f}};

    cout << "Tensor 1: (shape) = " << t1.shape_str() << " (size) = " << t1.size() << "\n";
    std::cout << t1 << "\n\n";

    cout << "Tensor 4: (shape) = " << t4.shape_str() << " (size) = " << t4.size() << "\n";
    std::cout << "t4[{0,1}]: " << t4[{0, 1}] << "\n";
    std::cout << t4 << "\n\n";



    cout << "---\n";
    Tensor t3 = t1 + t4;
    cout << "Tensor 3: (shape) = " << t3.shape_str() << " (size) = " << t3.size() << "\n";
    std::cout << t3 << "\n";

    Tensor t5 = t3 + 11.f;
    std::cout << t5 << "\n";

}

void timing(){
    Timer<> clock;

    clock.tic();
    Tensor::Seed(42);
    Tensor t1 = Tensor::Uniform(Shape{50, 28, 28});
    t1.requires_grad(true);
    Tensor::Seed(42);
    Tensor t2 = Tensor::Uniform(Shape{50, 28, 28});
    t2.requires_grad(true);

    Tensor::Seed(42);
    Tensor t3 = Tensor::Uniform(Shape{50, 28, 28});
    t3.requires_grad(true);
    clock.toc();
    std::cout << "Tensor creation: " << clock.duration().count() << "\n";


    clock.tic();
    Tensor t4 = t1 + t2;
    Tensor t5 = t3 * t4;
    t5 = Ops::tanh(t5);
    clock.toc();

    std::cout << "Operation: " << clock.duration<std::chrono::microseconds>().count() << "\n";

    clock.tic();
    t5.backward();
    clock.toc();
    std::cout << ".backward(): " << clock.duration<std::chrono::microseconds>().count() << "\n";


    Tensor tt = {1.f, 2.f, 3.f};
    tt = Ops::tanh(tt);
    cout << tt << "\n";

    Tensor::Seed(42);
    Tensor t_l = Tensor::Uniform(Shape{128, 784});
    Tensor::Seed(42);
    Tensor t_r = Tensor::Uniform(Shape{784, 32});
    clock.tic();
    Tensor dot = t_l.dot(t_r);
    clock.toc();
    std::cout << dot.shape() << "\n";
    std::cout << "Dot-prod: " << clock.duration<std::chrono::microseconds>().count() << "\n";

}


/**
 * Example model class for a MLP
*/
class MyModel : nn::Module{
public:
    nn::Linear l1;
    nn::Linear l2;
    nn::Linear l3;
    MyModel()
    : l1(nn::Linear(32, 16))
    , l2(nn::Linear(16, 2))
    , l3(nn::Linear(2, 1))
    {}

    Tensor forward(Tensor& x){
        Tensor a = l1(x);
        Tensor b = l2(a);
        Tensor c = l3(b);
        return c;
    }

    // call operator
    Tensor operator()(Tensor& x){
        return forward(x);
    }
};

void myMLP(){
    Tensor::Seed(42);
    MyModel net;

    Tensor input_ = Tensor::Uniform(0, 1, Shape{8, 32});
    Tensor out = net(input_);

    out.backward();

    //std::shared_ptr<Tensor> l1_w = net.l1.get_weights();
    //cout << *l1_w << "\n";

    // Apply the gradients to the tensors
    out.apply_grad(0.001f);


    cout << "end.\n";

}

void simple_example(){

    Tensor::Seed(42);
    Tensor a = Tensor::Normal(Shape{1});
    a.name = 'a';
    Tensor::Seed(43);
    Tensor b = Tensor::Normal(Shape{1});
    b.name = 'b';
    Tensor::Seed(44);
    Tensor c = Tensor::Normal(Shape{1});
    c.name = 'c';
    Tensor::Seed(45);
    Tensor d = Tensor::Normal(Shape{1});
    d.name = 'd';

    cout << a << " " << b << " " << c << " " << d << "\n";

    Tensor x (Shape{2000});
    x.name = 'x';
    x.requires_grad(false);
    Tensor y (Shape{2000});
    y.requires_grad(false);
    y.name = 'y';
    float v = -3.14f;
    float inter = (-1 * v * 2) / 2000;
    std::cout << inter << "\n";
    for(int i = 0; i < 2000; i++){
        x[i] = v;
        y[i] = std::sin(v);
        v = v + inter;
    }

    //y_pred = a + b * x + c * x ** 2 + d * x ** 3

    //y_pred = a + b * x + c * x ** 2 + d * x ** 3
    /*for(int i = 0; i < 2; i++){
        Tensor b_x = b * x;
        int temp = 3;
        Tensor x_pow_3 = Ops::power(x, temp);
        Tensor x_pow_2 = Ops::square(x);
        Tensor c_x = c * x_pow_2;
        Tensor d_x = d * x_pow_3;
        Tensor a_b = a + b_x;
        Tensor c_a = a_b + c_x;
        Tensor out = c_a + d_x;

        Tensor loss1 = out - y;
        Tensor loss2 = Ops::square(loss1);
        Tensor loss3 = loss2.sum();
        cout << loss1[0] << " " << loss1[1] << "\n";
        cout << out[0] << " " << out[1] << "\n";
        cout << y[0] << " " << y[1] << "\n";
        if ( i % 1 == 0 ){
            cout << "Epoch: " << i << " loss: " << loss3 << "\n";
        }
        loss3.backward();
        loss3.apply_grad(0.000001f);
        cout << "------\n";
    }*/

    for(int i = 0; i < 3000; i++ ){
        Tensor dx = Ops::power(x, 3); // 2000
        dx.name = "dx";
        Tensor dx2 = dx * d; // 2000
        dx2.name = "dx2";
        Tensor cx = Ops::power(x, 2); // 2000
        cx.name = "cx";
        Tensor cx2 = cx * c; // 2000
        cx2.name = "cx2";
        Tensor bx = b * x;
        bx.name = "bx";
        Tensor dc = dx2 + cx2;
        dc.name = "dc";
        Tensor dcb = dc + bx;
        dcb.name = "dcb";
        Tensor dcba = dcb + a;
        dcba.name = "dcba";

        Tensor loss1 = dcba - y; // 2000
        loss1.name = "loss1";
        Tensor loss2 = Ops::power(loss1, 2); // 2000
        loss2.name = "loss2";
        Tensor loss3 = loss2.sum(); // 1
        loss3.name = "loss3";

        if ( i % 2 == 0 ){
            cout << "Epoch: " << i << " loss: " << loss3 << "\n";
        }
        loss3.backward();
        /*cout << "a: " << a << *a.grad << "\n";
        cout << "b: " << b << *b.grad << "\n";
        cout << "c: " << c << *c.grad << "\n";
        cout << "d: " << d << *d.grad << "\n";
        cout << "\n";*/
        loss3.apply_grad(0.000001f);
    }

    cout << "a: " << a << "\n";
    cout << "b: " << b << "\n";
    cout << "c: " << c << "\n";
    cout << "d: " << d << "\n";

}


int main()
{
    //timing();
    //simd();
    //myMLP();
    simple_example();

    /*Tensor a = {0.5f};
    Tensor b  = Tensor::Uniform(Shape{2000});
    b = b.reshape(1, 2000);

    Tensor c = a * b;
    c.backward();

    cout << "a: " << a << " " << *a.grad << "\n";*/


    return 0;
}
