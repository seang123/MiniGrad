#include "Tensor.h"

#include <vector>
#include <algorithm>
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <x86intrin.h>
#include <smmintrin.h>

#define FastExpComputation

// Should be defined in math.h, but isn't showing as available
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

/**
 * A faster SIMD exp (e^) function
*/
static __m128 BetterFastExpSse (__m128 x)
{
  const __m128 a = _mm_set1_ps ((1 << 22) / float(M_LN2));  // to get exp(x/2)
  const __m128i b = _mm_set1_epi32 (127 * (1 << 23));       // NB: zero shift!
  __m128i r = _mm_cvtps_epi32 (_mm_mul_ps (a, x));
  __m128i s = _mm_add_epi32 (b, r);
  __m128i t = _mm_sub_epi32 (b, r);
  return _mm_div_ps (_mm_castsi128_ps (s), _mm_castsi128_ps (t));
}

/**
 * Extract a float from a __m128 vector by index [0, 4)
*/
template <unsigned i>
static float vectorGetByIndex( __m128 V ){
    V = _mm_shuffle_ps(V, V, _MM_SHUFFLE(i, i, i, i));
    return _mm_cvtss_f32(V);
}

namespace Ops{


template <typename T>
static float _tanh(T y){
#ifdef FastExpComputation
    // SIMD exp  ( 2x faster )
    // Has a small error at around 0.00X decimal points
    __m128 SSEa=_mm_load1_ps(&y);
    __m128 a = BetterFastExpSse(2 * SSEa);
    __m128 out = _mm_div_ps(a - 1.f, a + 1.f);
    //return vectorGetByIndex<0>(out);
    return _mm_cvtss_f32(out);
    /*float b = vectorGetByIndex<0>(a);
    return (b - 1) / (b + 1);*/
#else
    // Basic scalar arithmatic
    return (exp(2 * y) - 1 ) / (exp(2 * y) + 1);
#endif
}


/**
 * Computes tanh using SIMD approximation method 4 values at a time
*/
Tensor tanh_simd(Tensor& t){
    Tensor ret (t.shape());
    auto&& ret_data = ret.data();
    auto&& src_data = t.data();
    // Simply apply all
    for(size_t i = 0; i < t.size(); i+=4){
        __m128 vals = _mm_load_ps(&src_data[i]);
        __m128 a = BetterFastExpSse(2.f * vals);
        __m128 out = _mm_div_ps(a - 1.f, a+1.f);
        _mm_store_ps(&ret_data[i], out);
    }
    if(t.requires_grad()){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<tanh_op>(&t);
    }
    return ret;
}

/**
 * Computes tanh on a given tensor and returns a new tensor
*/
Tensor tanh(Tensor& t){

    if( t.size() < 4){
        // Single value SIMD - still faster than using math.h exp
        Tensor ret (t.shape());
        //ApplyOpSimple(ret, t, simple_tanh<float>);
        ApplyOpSimple(ret, t, _tanh<float>);
        if(t.requires_grad()){
            ret.requires_grad(true);
            ret.has_ctx = true;
            ret.ctx = std::make_shared<tanh_op>(&t);
        }
        return ret;
    } else{
        return tanh_simd(t);
    }
}

template <typename T>
float _exp(T y){
    return exp(y);
}

/**
 * Computes exp (e^) on a given tensor and returns a new tensor
*/
Tensor exp(Tensor& t){
    Tensor ret (t.shape());
#ifndef FastExpComputation
    ApplyOpSimple(ret, t, _exp<float>);
#else
    auto&& ret_data = ret.data();
    auto&& src_data = t.data();
    for(size_t i = 0; i < t.size(); i+=4){
        __m128 vals = _mm_load_ps(&src_data[i]);
        __m128 out = BetterFastExpSse(vals);
        _mm_store_ps(&ret_data[i], out);
    }
#endif

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

Tensor power(Tensor& t, float p){
    Tensor ret (t.shape());
    _pow<float> to_the_power(p);
    ApplyOpSimple(ret, t, to_the_power);
    if(t.requires_grad() == true){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<pow_op>(&t, p);
    }
    return ret;
}


Tensor power(Tensor& t, int p){
    return power(t, (float)p);
}


/**
 * Square the values in a tensor
*/
Tensor square(Tensor& t){
    /*Tensor ret = power(t, 2);
    if(t.requires_grad()){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<pow_op>(&t, 2);
    }
    return ret;*/
    //return power(t, 2);
    Tensor ret (t.shape());
    auto&& ret_data = ret.data();
    auto&& src_data = t.data();
    for(size_t i = 0; i < t.size(); i+=4){
        __m128 a = _mm_load_ps(&src_data[i]);
        __m128 b = _mm_load_ps(&src_data[i]);
        __m128 out = _mm_mul_ps(a, b);
        _mm_store_ps(&ret_data[i], out);
    }
    if(t.requires_grad() == true){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<pow_op>(&t, 2);
    }
    return ret;
}


}