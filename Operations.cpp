#include "Tensor.h"

#include <vector>
#include <algorithm>
#include <math.h>
#include <xmmintrin.h>

#define M_LN2 0.69314718055994530942

template <typename F>
static void runOp(int size, F op){
    for(int i = 0; i < size; i++){
        op(i);
    }
}

template <typename F>
inline void ApplyOpSimple(Tensor& ret, const Tensor& src, F op) {
    auto&& ret_data = ret.data();
    auto&& src_data = src.data();
    // Simply apply all
    for(size_t i = 0; i < src.size(); i++){
        ret_data[i] = op(src_data[i]);
    }
}

namespace Ops{

__m128 BetterFastExpSse (__m128 x)
{
  const __m128 a = _mm_set1_ps ((1 << 22) / float(M_LN2));  // to get exp(x/2)
  const __m128i b = _mm_set1_epi32 (127 * (1 << 23));       // NB: zero shift!
  __m128i r = _mm_cvtps_epi32 (_mm_mul_ps (a, x));
  __m128i s = _mm_add_epi32 (b, r);
  __m128i t = _mm_sub_epi32 (b, r);
  return _mm_div_ps (_mm_castsi128_ps (s), _mm_castsi128_ps (t));
}

template <unsigned i>
float vectorGetByIndex( __m128 V ){

    V = _mm_shuffle_ps(V, V, _MM_SHUFFLE(i, i, i, i));
    return _mm_cvtss_f32(V);
}

template <typename T>
static float _tanh(T y){
    // SIMD exp  ( 2x faster )
    // Has a small error at around 0.00X decimal points
    float t = 3.f;
    __m128 SSEa=_mm_load1_ps(&y);
    auto a = BetterFastExpSse(2 * SSEa);
    float b = vectorGetByIndex<0>(a);
    return (b - 1) / (b + 1);

    // Basic tanh eq
    //return (exp(2 * y) - 1 ) / (exp(2 * y) + 1);
}

template <typename T>
static float _exp(T y){
    return exp(y);
}

/**
 * Computes tanh on a given tensor and returns a new tensor
*/
Tensor tanh(Tensor& t){
    Tensor ret (t.shape());
    ApplyOpSimple(ret, t, _tanh<float>);

    if(t.requires_grad()){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<tanh_op>(&t);
    }

    return ret;
}

/**
 * Computes exp (e^) on a given tensor and returns a new tensor
*/
Tensor exp(Tensor& t){
    Tensor ret (t.shape());
    ApplyOpSimple(ret, t, _exp<float>);

    if(t.requires_grad()){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<exp_op>(&t);
    }

    return ret;
}

/**
 * A functor for computing the power of
*/
template <typename T>
class _pow{
    T p = 0;
public:
    _pow(T p){
        this->p = p;
    }
    float operator()(float b){
        return std::pow(b, p);
    }
};

/**
 * Takes the tensor to the power of some value
*/
template <typename T>
Tensor pow(Tensor& t, T p){
    Tensor ret (t.shape());
    _pow<T> to_the_power(p);
    ApplyOpSimple(ret, t, to_the_power);
    return ret;
}

/**
 * Square the values in a tensor
*/
Tensor square(Tensor& t){
    Tensor ret (t.shape());
    _pow<float> to_the_power(2);
    ApplyOpSimple(ret, t, to_the_power);
    return ret;
}

}