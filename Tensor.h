#ifndef _Tensor_H_
#define _Tensor_H_

#include <iostream>
#include <random>
#include <vector>
#include <memory>

#include "Substance.h"
#include "Iter.h"
#include "Ops.h"

using InitShape = std::initializer_list<int>; //allows braced-list initlization - Tenor t = {1,2,3};
using Shape = std::vector<int>;
using Index = std::vector<int>;
using Axis = std::vector<int>;

//class Iter;
class Op;
class Add_op;
class Sub_op;

struct Graph{
    std::set<Tensor*>visited;
    std::vector<Tensor*>nodes;
};


// ----------- Float initialisers ---------------
template <std::size_t D>
struct FloatListHelper {
    using type = std::initializer_list<typename FloatListHelper<D - 1>::type>;
};

template <>
struct FloatListHelper<0> {
    using type = std::initializer_list<float>;
};

template <std::size_t D>
using FloatList = typename FloatListHelper<D>::type;

// ------------ Tensor class -----------------
/**
 * Tensor class
 * A wrapper around a simple lazy buffer
 * Implements basic maths operations and backprop for gradient accumulation
*/
class Tensor{

private:
    std::shared_ptr<Substance> values;
    Tensor(std::shared_ptr<Substance>); // create tensor from substance pointer

    
    // Get begin-end of raw data array
    float* begin();
    float* end();
    const float* begin() const;
    const float* end() const;

    static std::random_device s_rand_seed;
    static std::mt19937 s_rand_engine;
    Graph ops_graph;

public:
    Tensor();
    Tensor(std::vector<float>, bool req_grad=true); // array, array_size
    Tensor(const InitShape& shape, bool req_grad=true);
    Tensor(const Shape& shape, bool req_grad=true);
    Tensor(const Shape& shape, float fill_v, bool req_grad=true);

    Tensor(const Tensor&); // deep copy another tensor obj
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(const Tensor&); // overwrite assignment operator
    Tensor& operator=(Tensor&&);
    ~Tensor();

    // tensors have gradients, buffers do not
    // gradients are themselves a tensor
    //Tensor* grad = nullptr;
    std::shared_ptr<Tensor> grad;
    bool requires_grad_ = true;
    void requires_grad(bool);
    bool requires_grad();
    bool requires_grad() const;

    std::string name;

    // Context - for computing the gradients
    std::shared_ptr<Op> ctx;
    bool has_ctx = false;

    static Tensor zeros(const Shape& shape); // Zero init tensor - parameter: shape
    static Tensor Ones(const Shape& shape);
    template <typename... S>
    static Tensor Ones(S... shape);

    static Tensor Arange(float stop);
    static Tensor Arange(float start, float stop, float step = 1.f);

    static void Seed();
    static void Seed(uint32_t);
    static Tensor Uniform(float, float, const Shape&);
    static Tensor Uniform(const Shape&);
    static Tensor Normal(float loc = 0.f, float scale = 1.f,
                          const Shape& shape = {1});
    static Tensor Normal(const Shape& shape);

    //Iter begin();
    //Iter end();

    // Sum two tensors - return new tensor
    Tensor sum();
    // Add a scalar to a tensor
    void add(int);
    Tensor add(const Tensor& lhs, const Tensor& rhs);

    Tensor square();
    Tensor square() const;

    // Reshape the tensor - dimensions listed in vector
    Tensor reshape(const Shape& shape) const;
    template <typename... S>
    Tensor reshape(S...) const;
    Tensor flatten() const;
    Tensor ravel() const;

    // Return a new copy of tensor
    Tensor copy() const;

    Tensor reduce_sum(int);

    Tensor dot(Tensor& other);

    // Overload index operator 
    float& operator[](const int);
    float& operator[](const size_t);
    float& operator[](const Index&) const;

    operator float() const;

    void print();
    size_t size() const;
    float* id() const;
    size_t ndim();
    size_t ndim() const;
    std::string shape_str();
    const Shape& shape() const;
    void fill(float);
    float* data();
    const float* data() const;
    bool empty() const;


    // --- backwards ---
    void backward();
    void apply_grad(float);
    //void deepwalk();
    Graph deepwalk();


    // Initialise from list
    Tensor(FloatList<0> init_list);
    Tensor(FloatList<1> init_list);
    Tensor(FloatList<2> init_list);
    Tensor(FloatList<3> init_list);
    Tensor(FloatList<4> init_list);
    Tensor(FloatList<5> init_list);
    Tensor(FloatList<6> init_list);
    Tensor(FloatList<7> init_list);
    Tensor(FloatList<8> init_list);
    Tensor(FloatList<9> init_list);


    static constexpr int DEFAULT_N_WORKERS = -1;
    static constexpr int DEFAULT_BATCH_SCALE = 4;
    static constexpr int DOT_CACHE_SCALE = 10;
    static int s_n_workers;
    static int s_batch_scale;
    static int GetNumWorkers();
    static void SetNumWorkers(int n_workers);
    static int GetBatchScale();
    static void SetBatchScale(int batch_scale);
};


Tensor Transpose(const Tensor&);

template <typename F>
void ApplyOpSimple(Tensor& dst, const Tensor& lhs, const Tensor& rhs, F op);

template <typename F>
inline void ApplyOpSimple(Tensor& ret, const Tensor& src, F op);

template <typename F>
void ApplyOpBroadcastImpl(float* ret_data,
                          const float* l_data,
                          const float* r_data,
                          const Shape& ret_shape, const int ret_size,
                          const std::vector<int>& l_steps,
                          const std::vector<int>& r_steps,
                          const size_t start_depth, const size_t n_depth,
                          const int ret_step, F op);
template <typename F>
void ApplyOpBroadcast(Tensor& ret, 
                    const Tensor& lhs, 
                    const Tensor& rhs, 
                    const size_t depth_offset, 
                    const int ret_step, 
                    F op);
//static std::vector<int> ComputeChildSizes(const Shape& shape);
//static size_t ReduceShapesBroadcast(Shape& ret_shape, Shape& l_shape,
//                                    Shape& r_shape, const size_t depth_offset);
//static Shape PadShape(const Shape& shape, size_t size);
Shape CheckBroadcastable(const Shape& lhs, const Shape& rhs);


template <typename F>
inline auto WrapOpForIter(F);

// --- Operators ---
Tensor operator+(const Tensor& lhs, float rhs);
Tensor operator-(const Tensor& lhs, float rhs);
Tensor operator*(const Tensor& lhs, float rhs);
Tensor operator/(const Tensor& lhs, float rhs);

Tensor operator+(float lhs, const Tensor& rhs);
Tensor operator-(float lhs, const Tensor& rhs);
Tensor operator*(float lhs, const Tensor& rhs);
Tensor operator/(float lhs, const Tensor& rhs);

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);

Tensor operator+(Tensor& lhs, Tensor& rhs);
Tensor operator-(Tensor& lhs, Tensor& rhs);
Tensor operator*(Tensor& lhs, Tensor& rhs);
Tensor operator*(Tensor& lhs, Tensor& rhs);
Tensor operator/(Tensor& lhs, Tensor& rhs);

Tensor operator+(float lhs, Tensor& rhs);
Tensor operator-(float lhs, Tensor& rhs);
Tensor operator*(float lhs, Tensor& rhs);
Tensor operator/(float lhs, Tensor& rhs);

Tensor operator+(Tensor& lhs, float rhs);
Tensor operator-(Tensor& lhs, float rhs);
Tensor operator*(Tensor& lhs, float rhs);
Tensor operator/(Tensor& lhs, float rhs);

// For temporary chaining -- ie. Tensor d = ((Tensor)a + (Tensor)b) + (Tensor)c as a+b will give a temporary rvalue-ref
Tensor operator+(Tensor&& lhs, Tensor&& rhs);
Tensor operator+(Tensor& lhs, Tensor&& rhs);
Tensor operator+(Tensor&& lhs, Tensor& rhs);

Tensor operator*(Tensor&& lhs, Tensor&& rhs);

Tensor operator==(const Tensor lhs, const Tensor& rhs);


// --- printing ---
std::ostream& operator<<(std::ostream&, const Tensor&);
/*static void OutputNdArray(std::ostream&, const Tensor&);
static void OutputArrayMultiDim(std::ostream& os,
                                const float* data,
                                const Shape& shape,
                                const std::vector<int>& child_sizes,
                                size_t depth);
static void OutputArrayLine(std::ostream& os, const float* data,
                            const int size);
*/

std::ostream& operator<<(std::ostream&, const Shape&);
//static void OutputShape(std::ostream&, const Shape&);



Tensor Sum(Tensor& x, const Axis& axes, bool keepdims);

#endif

