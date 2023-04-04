#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <immintrin.h>

#include "Tensor.h"
#include "Operations.h"

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
    Tensor t1 = Tensor::Uniform(Shape{50, 28, 28});
    t1.requires_grad(true);
    Tensor t2 = Tensor::Uniform(Shape{50, 28, 28});
    t2.requires_grad(true);

    Tensor t3 = Tensor::Uniform(Shape{50, 28, 28});
    t3.requires_grad(true);

    clock.toc();
    std::cout << "Tensor creation: " << clock.duration().count() << "\n";


    clock.tic();
    Tensor t4 = t1 + t2;
    Tensor t5 = t3 * t4;
    //t5 = Ops::tanh(t5);
    clock.toc();

    std::cout << "Operation: " << clock.duration<std::chrono::microseconds>().count() << "\n";

    clock.tic();
    t5.backward();
    clock.toc();
    std::cout << ".backward(): " << clock.duration<std::chrono::microseconds>().count() << "\n";


    Tensor tt = {1.f, 2.f, 3.f};
    tt = Ops::tanh(tt);
    cout << tt << "\n";

}

union U {
    __m128 v;
    float a[4];
};

template <int offsetRegs>
inline __m128 mul8(const float* p1, const float* p2){
    constexpr int lanes = offsetRegs * 4;
    const __m128 a = _mm_loadu_ps( p1 + lanes );
    const __m128 b = _mm_loadu_ps( p2 + lanes );
    return _mm_mul_ps(a, b);
};

void simd(){

    std::vector<float> a  = {1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<float> b  = {11.f, 22.f, 33.f, 44.f, 55.f};
    const float* p1 = a.data();
    const float* p2 = b.data();

    __m128 dot0 = mul8<0>(p1, p2);
    __m128 dot1 = mul8<1>(p1, p2);
     
    U u;
    u.v = dot0;
    std::cout << u.a[0] << "\n";
    std::cout << u.a[1] << "\n";
    std::cout << u.a[2] << "\n";
    std::cout << u.a[3] << "\n";

    u.v = dot1;
    std::cout << u.a[0] << "\n";

    std::cout << "-----\n";
    int size_ = 100000;
    int t1[size_];
    int t2[size_];
    for(int i = 0; i < size_; i++){
        t1[i] = i;
        t2[i] = i;
    }

    float out[size_];

    Timer<> clock;
    clock.tic();
    for(int i = 0; i < size_; i=i+4){
        __m128 a = _mm_set_ps( t1[i], t1[i+1], t1[i+2], t1[i+3] );
        __m128 b = _mm_set_ps( t2[i], t2[i+1], t2[i+2], t2[i+3] );
        __m128 c = _mm_add_ps(a, b);
        U u;
        u.v = c;
        out[i] = u.a[0];
        out[i+1] = u.a[1];
        out[i+2] = u.a[2];
        out[i+3] = u.a[3];
    }
    clock.toc();
    std::cout << "SIMD: " << clock.duration<std::chrono::microseconds>().count() << "\n";


}


int main()
{
    timing();
    //simd();

    return 0;
}
