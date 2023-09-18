#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <immintrin.h>
#include <math.h>

#include "Tensor.h"
#include "Operations.h"
#include "nn.h"

//#define TINYNDARRAY_IMPLEMENTATION
//#include "tinyndarray.h"
//using tinyndarray::NdArray;
using std::cout;

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

    std::cout << "----- tanh -----\n";

    Tensor tt = {1.f, 2.f, 3.f};
    //Tensor tt = Tensor::Uniform(Shape{2000});
    clock.tic();
    tt = Ops::tanh(tt);
    clock.toc();
    std::cout << "tanh single value simd: " << clock.duration<std::chrono::microseconds>().count() << "\n";

    tt = Tensor::Uniform(Shape{5000});
    float first = tt[0];
    clock.tic();
    tt = Ops::tanh(tt);
    clock.toc();
    std::cout << "tanh multi simd: " << clock.duration<std::chrono::microseconds>().count() << "\n";

    std::cout << "----- tanh end ------\n";

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

    std::vector<Tensor*> parameters(){
        std::vector<Tensor*> params;
        for( auto& m : {&l1, &l2, &l3}){
            auto module_params = m->parameters();
            params.insert(params.end(), module_params.begin(), module_params.end());
        }
        return params;
    }
};

void myMLP(){
    Tensor::Seed(42);
    MyModel net;

    Tensor input_ = Tensor::Uniform(0, 1, Shape{8, 32});
    Tensor out = net(input_);

    out.backward();


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

    cout << "start values: " << a << " " << b << " " << c << " " << d << "\n";

    Tensor x (Shape{2000});
    x.name = 'x';
    x.requires_grad(false);
    Tensor y (Shape{2000});
    y.requires_grad(false);
    y.name = 'y';
    float v = -3.14f;
    float inter = (-1 * v * 2) / 2000;
    for(int i = 0; i < 2000; i++){
        x[i] = v;
        y[i] = std::sin(v);
        v = v + inter;
    }


    Timer<> clock;
    clock.tic();
    int epochs = 2000;
    Tensor dx = Ops::power(x, 3);
    dx.requires_grad(false);
    Tensor cx = Ops::square(x); 
    cx.requires_grad(false);
    for(int i = 0; i < epochs; i++ ){
        //Tensor dx = Ops::power(x, 3); // 2000
        Tensor dx2 = dx * d; // 2000
        //Tensor cx = Ops::power(x, 2); // 2000
        Tensor cx2 = cx * c; // 2000
        Tensor bx = b * x;
        Tensor dc = dx2 + cx2;
        Tensor dcb = dc + bx;
        Tensor dcba = dcb + a;

        Tensor loss1 = dcba - y; // 2000
        Tensor loss2 = Ops::square(loss1); // 2000
        Tensor loss3 = loss2.sum(); // 1

        loss3.backward();
        loss3.apply_grad(0.000001f);
    }
    clock.toc();

    std::cout << "Epochs: " << epochs << "\n";
    std::cout << "time: " << clock.duration<std::chrono::milliseconds>().count() << " (ms)\n";

    cout << "a: " << a << "\n";
    cout << "b: " << b << "\n";
    cout << "c: " << c << "\n";
    cout << "d: " << d << "\n";
}

void compute_flops(long long time, Shape a, Shape b){
    // Compute flops for two tensors dot product operation
    // (n, m) * (m, p)  -> nm(2p - 1)
    float n_ops = 0.f;
    if (a.size() == 2){
        n_ops = a[0] * a[1] * (2 * b[1] - 1);
    } else if( a.size() == 3){
        n_ops = a[0] * (a[1] * a[2] * (2 * b[2] - 1));
    }
    float time_seconds = time / 1000000.f;
    float flops = (1 / time_seconds) * n_ops;
    std::cout << "flops: " << flops << "\n";
    std::cout << "Giga-flops: " << (flops * 1e-9) << "\n";
}

int main()
{
    //timing();
    //simd();
    //myMLP();
    //simple_example();


    /*
    Timer<> clock;
    Tensor tt = Tensor::Uniform(Shape{5000});
    float first = tt[0];
    clock.tic();
    tt = Ops::tanh(tt);
    clock.toc();
    std::cout << "tanh multi simd: " << clock.duration<std::chrono::microseconds>().count() << "\n";
    */

    Tensor::SetNumWorkers(2);
    Timer<> clock;
    clock.tic();
    Tensor t1 = Tensor::Uniform(Shape{128, 784});
    Tensor t2 = Tensor::Uniform(Shape{784, 4});
    clock.toc();
    std::cout << "dot prod init: " << clock.duration<std::chrono::microseconds>().count() << " (us)\n";

    clock.tic();
    Tensor t3 = t1.dot(t2);
    clock.toc();
    long long time = clock.duration<std::chrono::microseconds>().count();
    std::cout << "2D dot prod: " << clock.duration<std::chrono::microseconds>().count() << " (us)\n";
    compute_flops(time, Shape{128, 784}, Shape{128, 4});

    /// 3-D tensor 
    t1 = Tensor::Uniform(Shape{32, 128, 784});
    t2 = Tensor::Uniform(Shape{32, 784, 4});
    clock.tic();
    Tensor t31 = t1.dot(t2);
    clock.toc();
    time = clock.duration<std::chrono::microseconds>().count();

    /// Inner product
    t1 = Tensor::Uniform(Shape{2000});
    t2 = Tensor::Uniform(Shape{2000});
    clock.tic();
    Tensor t32 = t1.dot(t2);
    clock.toc();
    std::cout << "inner prod: " << clock.duration<std::chrono::microseconds>().count() << " (us)\n";

    return 0;
}