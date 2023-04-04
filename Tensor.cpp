#include <iostream>
#include <iterator>
#include <vector>
#include <memory>
#include <list>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <math.h>
#include <random>
#include <immintrin.h> // SIMD
#include <numeric> // std::accumulate
#include <thread> // std::thread
#include <atomic>

#include "Tensor.h"
#include "Substance.h"
#include "Ops.h"

// ----------------------- Forward declare ----------------------------

template <typename F>
void ApplyOpSimple(Tensor& dst, const Tensor& lhs, const Tensor& rhs, F op);

template <typename F>
inline void ApplyOpSimple(Tensor& ret, const Tensor& src, F op);

union U{
    __m128 v;
    float a[4];
};

template <typename F>
void runOp(int size, F op){
    for(int i = 0; i < size; i++){
        op(i);
    }
}

int Tensor::GetNumWorkers(){
    return s_n_workers;
}

void Tensor::SetNumWorkers(int n_workers) {
    s_n_workers = n_workers;
}

int Tensor::GetBatchScale() {
    return s_batch_scale;
}

void Tensor::SetBatchScale(int batch_scale) {
    s_batch_scale = batch_scale;
}

static void GetParallelParams(int size, int& n_workers, int& n_batch,
                              int& batch_size) {
    // Fetch the number of workers
    n_workers = Tensor::GetNumWorkers();
    if (n_workers <= 0) {
        n_workers = static_cast<int>(std::thread::hardware_concurrency());
    }
    // Compute batch size and it number
    n_batch = n_workers * Tensor::GetBatchScale();
    batch_size = size / n_batch + (size % n_batch ? 1 : 0);
    n_workers = std::min(n_workers, batch_size);
}

template <typename F, typename R>
float RunParallelWithReduce(int size, F op, R reduce, float init_v) {
    // Decide parallelization parameters
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetParallelParams(size, n_workers, n_batch, batch_size);

    if (n_workers <= 1) {
        // Single execution
        float v = init_v;
        for (int i = 0; i < size; i++) {
            // Operation with reduction
            v = reduce(v, op(i));
        }
        return v;
    } else {
        // Parallel execution
        std::atomic<int> next_batch(0);
        std::vector<std::thread> workers(static_cast<size_t>(n_workers));
        std::vector<float> results(workers.size());
        for (size_t t = 0; t < workers.size(); t++) {
            workers[t] = std::thread([ =, &next_batch, &results ]() noexcept {
                int batch_cnt = 0;
                float v = init_v;
                while ((batch_cnt = next_batch++) < n_batch) {
                    for (int i = 0; i < batch_size; i++) {
                        const int idx = batch_size * batch_cnt + i;
                        if (size <= idx) {
                            break;
                        }
                        // Operation with reduction
                        v = reduce(v, op(idx));
                    }
                }
                results[t] = v;
            });
        }
        for (auto&& worker : workers) {
            worker.join();
        }
        return std::accumulate(results.begin(), results.end(), init_v, reduce);
    }
}


// --------------------------------
/// ---=== Initilizers ===---
// --------------------------------

Tensor::~Tensor() = default;

// Create empty (size 0) array
Tensor::Tensor() : values(std::make_shared<Substance>()){};

// Create new tensor from pointer to substance class
Tensor::Tensor(std::shared_ptr<Substance> sub) : values(sub){};

// Shallow copy
//Tensor::Tensor(const Tensor& lhs) = default;
Tensor::Tensor(const Tensor& lhs) = default;

// Move
Tensor::Tensor(Tensor&& lhs) noexcept 
    : values(lhs.values)
    , grad(lhs.grad)
    , has_ctx(lhs.has_ctx)
    {
    has_ctx = lhs.has_ctx;
    ctx = lhs.ctx;
    requires_grad_ = lhs.requires_grad_;
};

// shallow copy
Tensor& Tensor::operator=(const Tensor& lhs) = default;

/** Move
 *  Data is moved from one tensor object to another (efficiently)
 * Old object may be empty afterwards (will probably be destroyed soon)
*/
Tensor& Tensor::operator=(Tensor&& lhs){
    std::cout << "operator=(Tensor&&)\n";
    values = std::move(lhs.values);
    return *this;
}

//Tensor::Tensor(const InitShape& shape) : Tensor(Shape(shape)){};

/**
 * Init tensor from a vector describing its shape
*/
Tensor::Tensor(const Shape& shape, bool req_grad){
    size_t size = 1;
    for (auto&& s : shape){
        if (s < 0){
            throw std::runtime_error("Invalid shape format (neg)");
        }
        size *= static_cast<size_t>(s);
    }
    values = std::make_shared<Substance>(size, shape);
    this->requires_grad_ = req_grad;
}

Tensor::Tensor(const Shape& shape, float fill_v, bool req_grad) : Tensor(shape, req_grad) {
    fill(fill_v);
    //this->requires_grad_ = req_grad;
}

/**
 * Init tensor form 1-D vector
*/
Tensor::Tensor(std::vector<float> vec, bool req_grad){
    //throw std::runtime_error("Not implemented");
    values = std::make_shared<Substance>(vec.size(), std::vector<int>(vec.size()));
    for(size_t i = 0; i < vec.size(); i++){
        values->vals.get()[i] = vec[i];
    }
    this->requires_grad_ = req_grad;
};

// ------------ Float list initializers -------------


template <typename FList>
std::list<int> CheckFListShapeImpl(const FList& init_list){
    if (init_list.size() == 0){
        return {};
    }
    // Check that all sub-lists have equal shape
    auto itr = init_list.begin();  // get first element
    auto shape = CheckFListShapeImpl(*itr);  // recursively check sub-elements
    for (size_t i = 0; i < init_list.size(); i++, itr++){
        if (shape != CheckFListShapeImpl(*itr)){
            throw std::runtime_error("Initializing shape is invalid. Dimensions are not equal.");
        }
    }
    // Total shape of children
    shape.push_front(static_cast<int>(init_list.size()));
    return shape;
}

/**
 * Base case: if init_list holds only floats
*/
template <>
inline std::list<int> CheckFListShapeImpl(const FloatList<0>& init_list){
    return {static_cast<int>(init_list.size())};
}

template <typename FList>
void CopyFListElemsImpl(const FList& init_list, float*& data){
    // copy sequentially
    for (auto itr = init_list.begin(); itr != init_list.end(); itr++){
        CopyFListElemsImpl(*itr, data);
    }
}

/**
 * Returns the shape of a nested initialiser list
*/
template <typename FList>
Shape CheckFListShape(const FList& init_list){
    // Check and get the shape of nested initialiser
    const std::list<int>& shape = CheckFListShapeImpl(init_list);
    // Cast to vector
    return Shape(shape.begin(), shape.end());
}

template <>
void CopyFListElemsImpl(const FloatList<0>& init_list, float*&data){
    // Copy sequentially 
    for(auto&& v : init_list){
        *(data++) = v;
    }
}

template <typename FList>
void CopyFListElems(const FList& init_list, float* data){
    CopyFListElemsImpl(init_list, data);
}

Tensor::Tensor(FloatList<0> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<1> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<2> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<3> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<4> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<5> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<6> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<7> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<8> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}

Tensor::Tensor(FloatList<9> init_list) : Tensor(CheckFListShape(init_list)){
    CopyFListElems(init_list, values->vals.get());
}


// ---------------- Tensor init of 1's -------------------

Tensor Tensor::Ones(const Shape& shape){
    return Tensor(shape, 0.f);
}

/**
 * Tensor t = Tensor::Ones(1, 2, 3, 4);
 * t.shape() == 1x2x3x4
*/
template <typename... S>
Tensor Tensor::Ones(S... shape){
    return Ones({shape...});
}

// ---------------- Random data init ------------------

std::random_device Tensor::s_rand_seed;
std::mt19937 Tensor::s_rand_engine(s_rand_seed());
int Tensor::s_n_workers = Tensor::DEFAULT_N_WORKERS;
int Tensor::s_batch_scale = Tensor::DEFAULT_BATCH_SCALE;

void Tensor::Seed(){
    s_rand_engine = std::mt19937(s_rand_seed());
}

void Tensor::Seed(uint32_t seed){
    s_rand_engine = std::mt19937(seed);
}

template <typename D, typename R>
Tensor CreateRandomArray(const Shape& shape, D&& dist, R&& rand_engine){
    Tensor ret(shape);
    ApplyOpSimple(ret, [&]() { return static_cast<float>(dist(rand_engine));});
    return ret;
}

Tensor Tensor::Uniform(float low, float high, const Shape& shape){
    std::uniform_real_distribution<> dist(low, high);
    return CreateRandomArray(shape, dist, s_rand_engine);
}

Tensor Tensor::Uniform(const Shape& shape){
    return Uniform(0.f, 1.f, shape);
}

// ------------------------
/// --- Cast --------------
// ------------------------

Tensor::operator float() const {
    if (values->size != 1){
        std::cout << "float() values->size = " << values->size << "\n";
        throw std::runtime_error("Only size 1 array can be converted to float");
    }
    return *(values->vals.get());
}

// -------------------------
// ----- Index methods -----
// -------------------------

// shape: 3, 2, 5
// index: 0, 0, 1

float& Tensor::operator[](const Index& index) const {
    const Shape& shape = values->shape;
    if (index.size() != shape.size()){
        throw std::runtime_error("Invalid index size");
    }
    int i = 0;
    for(size_t d = 0; d < index.size(); d++){
        i *= shape[d];
        // Allow for negative indexing
        const int p_idx = (index[d] >= 0) ? index[d] : shape[d] + index[d];
        i += p_idx;
    }
    return *(values->vals.get() + i);
}

// -------------------------

// ------------- Reshape --------------


// ------------- Basic getters & setters --------------

float& Tensor::operator[](const int index){
    if((size_t)index > (values->size - 1)){
        throw std::runtime_error("Index out of range.");
    }
    return values->vals.get()[index];
}

/**
 * Total length of raw data array
*/
size_t Tensor::size() const {
    return values->size;
}

/**
 * Return pointer to first data element
*/
float* Tensor::id() const{
    return values->vals.get();
}

/** 
 * Return the tensors shape as a string
 */
std::string Tensor::shape_str(){
    std::stringstream result;
    std::copy(
        values->shape.begin(), 
        values->shape.end(), 
        std::ostream_iterator<int>(result, " "));
    return result.str();
}


const Shape& Tensor::shape() const {
    return values->shape;
}


bool Tensor::empty() const {
    return values->size == 0;
}

/**
 * Number of dimensions
*/
size_t Tensor::ndim() {
    return values->shape.size();
}

size_t Tensor::ndim() const{
    return values->shape.size();
}

/** Helper
 * Fill the data array with some value
*/
void fillN(float* iter, const int n, float fill_v){
    runOp(n, [&](int i){ iter[i] = fill_v; });
};

/** 
 * Fill the data array with some value
*/
void Tensor::fill(float v){
    fillN(values->vals.get(), static_cast<int>(values->size), v);
}

/*Iter Tensor::begin() {
    return Iter(values->vals.get());
}

Iter Tensor::end() {
    return Iter(values->vals.get() + values->size);
}*/

float* Tensor::begin(){
    return values->begin();
}

float* Tensor::end(){
    return values->end();
}

const float* Tensor::begin() const{
    return values->begin();
}

const float* Tensor::end() const{
    return values->end();
}

float* Tensor::data(){
    return begin();
}

const float* Tensor::data() const{
    return begin();
}

/**
 * Set the requires_grad flag of a tensor (needed when doing floatlist init)
*/
void Tensor::requires_grad(bool req_grad){
    this->requires_grad_ = req_grad;
}

/**
 * Return bool indicating if gradients are being accumulated for tensor
*/
bool Tensor::requires_grad(){
    return this->requires_grad_;
}
bool Tensor::requires_grad() const {
    return this->requires_grad_;
}

// --------------------- Applying an operations ------------------
/**
 * Apply a simple math operation across two tensors that have the same shape
 * Result of the operation is stored in dst tensor
*/

#ifdef SIMD_methods
template <typename F>
void ApplyOpSimple(Tensor& dst, const Tensor& lhs, const Tensor& rhs, F op) {
    float* dst_ = dst.id();
    float* lhs_ = lhs.id();
    float* rhs_ = rhs.id();
    for(size_t i = 0; i < dst.size(); i = i + 4){
        __m128 a = op(lhs_, rhs_);
        U u;
        u.v = a;
        dst_[i] = u.a[3];
        dst_[i+1] = u.a[2];
        dst_[i+2] = u.a[1];
        dst_[i+3] = u.a[0];
    }
}
#else
template <typename F>
void ApplyOpSimple(Tensor& dst, const Tensor& lhs, const Tensor& rhs, F op) {
    float* dst_ = dst.id();
    float* lhs_ = lhs.id();
    float* rhs_ = rhs.id();
    for(size_t i = 0; i < lhs.size(); i++){
        //dst_[i] = lhs_[i] + rhs_[i];
        dst_[i] = op(lhs_[i], rhs_[i]);
    }
}
#endif

/**
 * ApplyOpSimple with only 1 source tensor
*/
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
 * ApplyOpSimple with only a return tensor and a function
*/
template <typename F>
inline void ApplyOpSimple(Tensor& ret, F op) {
    auto&& ret_data = ret.data();
    for(size_t i = 0; i < ret.size(); i++){
        ret_data[i] = op();
    }
}

// --------------------------------------------------------------------

// --------------------- Broadcasting tensors -----------------------
/**
 * Return a new shape which allows two tensors have an arithmetic operation applied to them
*/
Shape CheckBroadcastable(const Shape& lhs, const Shape& rhs){
    if(lhs.size() < rhs.size()){
        // Assume that left tensor is deeper than right tensor
        return CheckBroadcastable(rhs, lhs);
    }
    if(rhs.size() == 0 || (rhs.size() == 1 && rhs[0] == 0)){
        throw std::runtime_error("Broadcasting empty array");
    }

    // lhs: [3, 4, 3, 2]
    // rhs: [3, 4, 1]

    Shape shape(lhs.size());
    // Difference in depth
    size_t r_offset = lhs.size() - rhs.size();
    for( size_t i = 0; i < lhs.size(); i++ ){
        if(i < r_offset){
            shape[i] = lhs[i];
        }
        else{
            const int l = lhs[i];
            const int r = rhs[i - r_offset];
            if(l == r){
                shape[i] = l; // no broadcast
            } else if (l == 1){
                shape[i] = r; // left broadcast
            } else if (r == 1){
                shape[i] = l; // right broadcast
            } else{
                std::stringstream ss;
                ss << "Non operatable shapes ";
                ss << "(" << lhs << " & " << rhs << ")";
                throw std::runtime_error(ss.str());
            }
        }
    }
    return shape;
}

/**
 * Return a new shape padded with extra dimensions of size 1
*/
static Shape PadShape(const Shape& shape, size_t size) {
    if (size < shape.size()) {
        throw std::runtime_error("Invalid shape to pad");
    }
    const size_t n_pad = size - shape.size();
    Shape ret_shape;
    ret_shape.reserve(size);
    ret_shape.resize(n_pad, 1);                                     // Fill by 1
    ret_shape.insert(ret_shape.end(), shape.begin(), shape.end());  // Concat
    return ret_shape;
}

/**
 * Reduce the Shapes of tensors that are being operated on
*/
static size_t ReduceShapesBroadcast(Shape& ret_shape, Shape& l_shape,
                                    Shape& r_shape, const size_t depth_offset) {
    // Require `ret_shape.size() == l_shape.size() == r_shape.size()`

    // Remove meaningless dimensions.
    Shape ret_shape_cleaned, l_shape_cleaned, r_shape_cleaned;
    int size_pool = 1;
    size_t depth = 0;
    for (; depth < ret_shape.size() - depth_offset; depth++) {
        if (l_shape[depth] == r_shape[depth]) {
            // Store
            size_pool *= l_shape[depth];
        } else {
            // Pop
            if (size_pool != 1) {
                ret_shape_cleaned.push_back(size_pool);
                l_shape_cleaned.push_back(size_pool);
                r_shape_cleaned.push_back(size_pool);
                size_pool = 1;
            }
            // Through current dimension
            ret_shape_cleaned.push_back(ret_shape[depth]);
            l_shape_cleaned.push_back(l_shape[depth]);
            r_shape_cleaned.push_back(r_shape[depth]);
        }
    }
    // Pop
    if (size_pool != 1 || ret_shape_cleaned.size() == 0) {
        ret_shape_cleaned.push_back(size_pool);
        l_shape_cleaned.push_back(size_pool);
        r_shape_cleaned.push_back(size_pool);
    }
    // Store actual depth count
    const size_t n_depth = ret_shape_cleaned.size();
    // Pass through included in `depth_offset`.
    for (; depth < ret_shape.size(); depth++) {
        ret_shape_cleaned.push_back(ret_shape[depth]);
        l_shape_cleaned.push_back(l_shape[depth]);
        r_shape_cleaned.push_back(r_shape[depth]);
    }
    // Return
    ret_shape = std::move(ret_shape_cleaned);
    l_shape = std::move(l_shape_cleaned);
    r_shape = std::move(r_shape_cleaned);
    return n_depth;
}

/**
 * Size of each dimensions sub-dimensions
 * shape = [3, 4, 2] 
 * child_sizes = [24, 8, 2]
*/
static std::vector<int> ComputeChildSizes(const Shape& shape) {
    const size_t n_shape = shape.size();
    if (n_shape == 0) {
        return {};
    }
    // Compute child sizes from back (the number of children for each dimension)
    std::vector<int> child_sizes(n_shape, 1);
    int size = 1;
    for (size_t depth = n_shape - 1; 0 < depth; depth--) {
        child_sizes[depth] = size;
        size *= shape[depth];
    }
    child_sizes[0] = size;
    return child_sizes;
}

template <typename F>
void ApplyOpBroadcastImpl(float* ret_data,
                          const float* l_data,
                          const float* r_data,
                          const Shape& ret_shape, const int ret_size,
                          const std::vector<int>& l_steps,
                          const std::vector<int>& r_steps,
                          const size_t start_depth, const size_t n_depth,
                          const int ret_step, F op) {
    // Create stacks and counter
    std::vector<int> ret_cnts(n_depth);
    std::vector<int> l_idx_stack(n_depth), r_idx_stack(n_depth);
    size_t depth = start_depth;
    int l_idx = 0;
    int r_idx = 0;

    for (int ret_idx = 0; ret_idx < ret_size; ret_idx += ret_step) {
        // Go down
        for (; depth < n_depth; depth++) {
            l_idx_stack[depth] = l_idx;  // Push stack
            r_idx_stack[depth] = r_idx;
        }

        // Operate
        op(ret_data + ret_idx, l_data + l_idx, r_data + r_idx);

        // Go up and count
        for (; start_depth < depth; depth--) {
            const size_t prev_d = depth - 1;
            ret_cnts[prev_d]++;        // Count up
            l_idx += l_steps[prev_d];  // Forward index
            r_idx += r_steps[prev_d];
            if (ret_cnts[prev_d] < ret_shape[prev_d]) {
                break;  // Continue normally
            }
            // Go upper depth
            ret_cnts[prev_d] = 0;         // Clear count
            l_idx = l_idx_stack[prev_d];  // Pop stack
            r_idx = r_idx_stack[prev_d];
        }
    }
}


/**
 * Apply an operation to two tensors after broadcasting them to a matching shape
*/
template <typename F>
void ApplyOpBroadcast(Tensor& ret, 
                    const Tensor& lhs, 
                    const Tensor& rhs, 
                    const size_t depth_offset, 
                    const int ret_step, 
                    F op){
    Shape ret_shape = ret.shape();

    Shape l_shape = PadShape(lhs.shape(), ret_shape.size());
    Shape r_shape = PadShape(rhs.shape(), ret_shape.size());

    const size_t n_depth = ReduceShapesBroadcast(ret_shape, l_shape, r_shape, depth_offset);

    const std::vector<int>& ret_child_sizes = ComputeChildSizes(ret_shape);
    const std::vector<int>& l_child_sizes = ComputeChildSizes(l_shape);
    const std::vector<int>& r_child_sizes = ComputeChildSizes(r_shape);

    std::vector<int> l_steps, r_steps;
    l_steps.reserve(n_depth);
    r_steps.reserve(n_depth);
    for( size_t depth = 0; depth < n_depth; depth++){
        const int& l_s = l_shape[depth];
        const int& r_s = r_shape[depth];
        const int l_step = (l_s == r_s || r_s == 1) ? l_child_sizes[depth] : 0;
        const int r_step = (l_s == r_s || l_s == 1) ? r_child_sizes[depth] : 0;
        l_steps.push_back(l_step);
        r_steps.push_back(r_step);
    }

    ApplyOpBroadcastImpl(ret.data(), lhs.data(), rhs.data(), ret_shape,
                        static_cast<int>(ret.size()), 
                        l_steps, r_steps, 0, n_depth, ret_step, op);
}

/**
 * Wrap an operator to be applied to broadcasted array
*/
template <typename F>
inline auto WrapOpForIter(F op){
    return [op](float* o, const float* l, const float* r){
        *o = op(*l, *r);
    };
}

// -------------------------------- Math operators ----------------------------

template <typename F>
Tensor ApplyDualOp(const Tensor& lhs, const Tensor& rhs, F op) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        Tensor ret(lhs.shape());
        if(lhs.requires_grad() || rhs.requires_grad()){ 
            ret.requires_grad(true); 
        };
        // Simply apply all
        ApplyOpSimple(ret, lhs, rhs, op);
        return ret;
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Apply broadcast
        Tensor ret(ret_shape);
        if(lhs.requires_grad() || rhs.requires_grad()){ 
            ret.requires_grad(true); 
        };
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(op));
        return ret;
    }
}

/* --------- Addition ------------ */

class fast_add_{
public:
    __m128 operator()(const float* lhs, const float* rhs){
        __m128 aa = _mm_set_ps(*(lhs), *(lhs+1), *(lhs+2), *(lhs+3));
        __m128 bb = _mm_set_ps(*(rhs), *(rhs+1), *(rhs+2), *(rhs+3));
        __m128 out = _mm_add_ps(aa, bb);
        return out;
    }

    const float operator()(const float& lhs, const float& rhs) const {
        __m128 aa = _mm_set_ps(lhs, lhs, lhs, lhs);
        __m128 bb = _mm_set_ps(rhs, rhs, rhs, rhs);
        __m128 out = _mm_add_ps(aa, bb);
        U u;
        u.v = out;
        return u.a[0];
    }
    float operator()(const float& lhs, const float& rhs){
        __m128 aa = _mm_set_ps(lhs, lhs, lhs, lhs);
        __m128 bb = _mm_set_ps(rhs, rhs, rhs, rhs);
        __m128 out = _mm_add_ps(aa, bb);
        U u;
        u.v = out;
        return u.a[0];
    }
};


Tensor operator+(const Tensor& lhs, const Tensor& rhs){
    if(lhs.shape() == rhs.shape()){
        Tensor ret(lhs.shape());
        // simple sum
        ApplyOpSimple(ret, lhs, rhs, std::plus<float>());
        return ret;
    }
    else{
        // Check if broadcastable
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        Tensor ret(ret_shape);
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(std::plus<float>()));
        return ret;
    }
}

/*
Tensor operator+(const Tensor& lhs, float rhs){
    // Technically operator+(const Tensor&, Tensor&&) but degrades to const lvalue reference
    // TODO: this is wrong -- rhs.shape should equal lhs.shape
    return lhs + Tensor(Shape({1}), rhs);
}

Tensor operator+(float lhs, const Tensor& rhs){
    // TODO: this is wrong -- lhs.shape should equal rhs.shape
    return Tensor(Shape({1}), lhs) + rhs;
}
*/

Tensor operator+(float lhs, Tensor& rhs){
    // Have to do it in 2 steps because I don't currently support rvalue references
    // Otherwise, could write: return Tensor(lhs) - rhs
    Tensor lhs_ (rhs.shape(), lhs);
    return lhs_ + rhs;
}

Tensor operator+(Tensor& lhs, float rhs){
    return rhs + lhs;
}


//* Tensors which should accumulate gradients with their operations need to be non-const
/**
 * Runs addition operation on two tensors while also setting the context variable for backprop
*/
Tensor add(Tensor& lhs, Tensor& rhs){
    Tensor ret = ApplyDualOp(lhs, rhs, std::plus<float>());
    if(lhs.requires_grad() || rhs.requires_grad()){ 
        ret.requires_grad(true); 
        //* The output stores referenecs to the parent tensors as well as what operation produced it
        ret.ctx = std::make_shared<Add_op>(&lhs, &rhs);
        ret.has_ctx = true;
    };
    return ret;
}

Tensor operator+(Tensor& lhs, Tensor& rhs){
    return add(lhs, rhs);
}


// ----------- Subtraction ----------

/*
Tensor sub(const Tensor& lhs, const Tensor& rhs){
    Tensor ret = ApplyDualOp(lhs, rhs, std::minus<float>());
    if(lhs.requires_grad() || rhs.requires_grad()){
        ret.requires_grad(true);
        Tensor lhs_ = lhs;
        Tensor rhs_ = rhs;
        ret.ctx = std::make_shared<Sub_op>(&lhs_, &rhs_);
        ret.has_ctx = true;
    }
    return ret;
}
*/
class fast_sub_{
public:
    __m128 operator()(const float* lhs, const float* rhs){
        __m128 aa = _mm_set_ps(*(lhs), *(lhs+1), *(lhs+2), *(lhs+3));
        __m128 bb = _mm_set_ps(*(rhs), *(rhs+1), *(rhs+2), *(rhs+3));
        __m128 out = _mm_sub_ps(aa, bb);
        return out;
    }

    const float operator()(const float& lhs, const float& rhs) const {
        __m128 aa = _mm_set_ps(lhs, lhs, lhs, lhs);
        __m128 bb = _mm_set_ps(rhs, rhs, rhs, rhs);
        __m128 out = _mm_mul_ps(aa, bb);
        U u;
        u.v = out;
        return u.a[0];
    }
    float operator()(const float& lhs, const float& rhs){
        __m128 aa = _mm_set_ps(lhs, lhs, lhs, lhs);
        __m128 bb = _mm_set_ps(rhs, rhs, rhs, rhs);
        __m128 out = _mm_mul_ps(aa, bb);
        U u;
        u.v = out;
    }
};


Tensor sub(Tensor& lhs, Tensor& rhs){
    Tensor ret = ApplyDualOp(lhs, rhs, std::minus<float>());
    if(lhs.requires_grad() || rhs.requires_grad()){
        ret.requires_grad(true);
        ret.ctx = std::make_shared<Sub_op>(&lhs, &rhs);
        ret.has_ctx = true;
    }
    return ret;
}


Tensor operator-(float lhs, Tensor& rhs){
    // Have to do it in 2 steps because I don't currently support rvalue references
    // Otherwise, could write: return Tensor(lhs) - rhs
    Tensor lhs_ (rhs.shape(), lhs);
    return lhs_ - rhs;
}

Tensor operator-(Tensor& lhs, float rhs){
    Tensor rhs_ (lhs.shape(), rhs);
    return lhs - rhs_;
}

Tensor operator-(Tensor& lhs, Tensor& rhs){
    return sub(lhs, rhs);
}

/*
Tensor operator-(float lhs, const Tensor& rhs){
    const Tensor lhs_ (rhs.shape(), lhs);
    return sub(lhs_, rhs);
}
*/

// ---------- Multiplication -----------

class fast_mul_{
public:
    __m128 operator()(const float* lhs, const float* rhs){
        __m128 aa = _mm_set_ps(*(lhs), *(lhs+1), *(lhs+2), *(lhs+3));
        __m128 bb = _mm_set_ps(*(rhs), *(rhs+1), *(rhs+2), *(rhs+3));
        __m128 out = _mm_mul_ps(aa, bb);
        return out;
    }

    const float operator()(const float& lhs, const float& rhs) const {
        __m128 aa = _mm_set_ps(lhs, lhs, lhs, lhs);
        __m128 bb = _mm_set_ps(rhs, rhs, rhs, rhs);
        __m128 out = _mm_mul_ps(aa, bb);
        U u;
        u.v = out;
        return u.a[0];
    }
    float operator()(const float& lhs, const float& rhs){
        __m128 aa = _mm_set_ps(lhs, lhs, lhs, lhs);
        __m128 bb = _mm_set_ps(rhs, rhs, rhs, rhs);
        __m128 out = _mm_mul_ps(aa, bb);
        U u;
        u.v = out;
        return u.a[0];
    }
};

Tensor operator*(const Tensor& lhs, const Tensor& rhs){
    if(lhs.shape() == rhs.shape()){
        Tensor ret(lhs.shape());
        // simple sum
        ApplyOpSimple(ret, lhs, rhs, std::multiplies<float>());
        return ret;
    }
    else{
        // Check if broadcastable
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        Tensor ret(ret_shape);
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(std::multiplies<float>()));
        return ret;
    }
}

Tensor operator*(const Tensor& lhs, float rhs){
    // Technically operator+(const Tensor&, Tensor&&) but degrades to const lvalue reference
    return lhs * Tensor(Shape({1}), rhs);
}

Tensor operator*(float lhs, const Tensor& rhs){
    return Tensor(Shape({1}), lhs) * rhs;
}

Tensor mul(Tensor& lhs, Tensor& rhs){
    Tensor ret = ApplyDualOp(lhs, rhs, std::multiplies<float>());
    if(lhs.requires_grad() || rhs.requires_grad()){
        ret.requires_grad(true);
        ret.ctx = std::make_shared<Mul_op>(&lhs, &rhs);
        ret.has_ctx = true;
    }
    return ret;
}

Tensor operator*(Tensor& lhs, Tensor& rhs){
    return mul(lhs, rhs);
}

// ---------- Power -----------

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
 * Square (^2) a tensor and return a new tensor
*/
Tensor Tensor::square(){
    Tensor ret (this->shape());
    _pow<int> _square(2);
    ApplyOpSimple(ret, *this, _square);
    return ret;
}

/**
 * Square a const tensor and return a new tensor
*/
Tensor Tensor::square() const {
    Tensor ret (this->shape());
    _pow<int> _square(2);
    ApplyOpSimple(ret, *this, _square);
    return ret;
}


// --------------------------- Dot product --------------------------------

/**
 * Dot product for 1-d array
*/
static Tensor DotTensor1d(const Tensor& lhs, const Tensor& rhs) {
    const size_t size = lhs.size();
    if (size != rhs.size()) {
        throw std::runtime_error("Invalid size for inner product of 1D");
    }
    // Inner product of vectors
    auto&& l_data = lhs.data();
    auto&& r_data = rhs.data();
    auto&& op = [&](int i) { return l_data[i] * r_data[i]; };
    // Run in parallel
    const float sum = RunParallelWithReduce(static_cast<int>(size), op,
                                            std::plus<float>(), 0.f);
    return {sum};
}

/**
 * Computes dot product along columns
*/
static void DotTensor1d2dImplColMajor(
    float* ret_data,
    const float* l_data,
    const float* r_data,
    const int n_col, const int n_contract
){
    std::fill_n(ret_data, n_col, 0.f);
    int r_idx = 0;
    for(int l_idx = 0; l_idx < n_contract; l_idx++){
        for(int col_cnt = 0; col_cnt < n_col; col_cnt++){
            ret_data[col_cnt] += l_data[l_idx] * r_data[r_idx];
            r_idx++;
        }
    }
}

/**
 * Computes dot product along rows
*/
static void DotTensor1d2dImplRowMajor(
    float* ret_data,
    const float* l_data,
    const float* r_data,
    const int n_col, const int n_contract
){
    for( int col_cnt = 0; col_cnt < n_col; col_cnt++ ){
        float sum = 0.f;
        int r_idx = col_cnt;
        for( int l_idx = 0; l_idx < n_contract; l_idx++ ){
            sum += l_data[l_idx] * r_data[r_idx];
            r_idx += n_col;
        }
        ret_data[col_cnt] = sum;
    }
}

/**
 * Returns either a column-major or row-major dot product function
*/
static auto SelectDot1d2dOp(const Shape& l_shape, const Shape& r_shape){

    const int left = l_shape.end()[-1];
    const int right = r_shape.end()[-2] * r_shape.end()[-1];
    if( left * Tensor::DOT_CACHE_SCALE < right){
        return DotTensor1d2dImplColMajor;
    } else{
        return DotTensor1d2dImplRowMajor;
    }
}

/**
 * Compute dot product of 2 tensors
*/
template <typename F1d2d>
void DotTensorNdMdImpl(
    float* ret_data, 
    const float* l_data,
    const float* r_data,
    const int n_l, const int n_r,
    const int ret_step, const int l_step, const int r_step, F1d2d op_1d2d
){
    const int& n_contract = l_step;
    const int& n_col = ret_step;
    const int& ret_idx_base = n_r;

    int l_idx = 0;
    int ret_idx0 = 0;
    for (int l_cnt = 0; l_cnt < n_l; l_cnt++) {
        int r_idx = 0;
        int ret_idx = ret_idx0 * ret_step;
        for (int r_cnt = 0; r_cnt < n_r; r_cnt++) {
            op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx, n_col,
                    n_contract);
            r_idx += r_step;
            ret_idx += ret_step;
        }
        l_idx += l_step;
        ret_idx0 += ret_idx_base;
    }
}

/**
 * Compute dot product for multidimensional tensors
*/
static Tensor DotTensorNdMd(const Tensor& lhs, const Tensor& rhs){
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();

    const int n_contract = l_shape.end()[-1];
    if (n_contract != r_shape.end()[-2]){
        throw std::runtime_error("Invalid shape for dot product");
    }

    // Same as input minus last dim
    Shape ret_shape(l_shape.begin(), l_shape.end() - 1);
    ret_shape.insert(ret_shape.end(), r_shape.begin(), r_shape.end() - 2);
    ret_shape.push_back(r_shape.back());

    Tensor ret(ret_shape);

    const int ret_step = r_shape.end()[-1]; // size of rhs last dim
    const int l_step = n_contract; // size of steps to take on the lhs 
    const int r_step = n_contract * ret_step;  // size of steps to take on the rhs

    const int n_l = static_cast<int>(lhs.size()) / l_step; // nr. of steps we'll have to do on lhs
    const int n_r = static_cast<int>(rhs.size()) / r_step; // nr. of steps we'll have to do on rhs

    DotTensorNdMdImpl(ret.data(), lhs.data(), rhs.data(), n_l, n_r, ret_step, l_step, r_step, SelectDot1d2dOp(l_shape, r_shape));
    return ret;
}

static Tensor DotTensor(const Tensor& lhs, const Tensor& rhs) {
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    if (lhs.size() == 0 || rhs.size() == 0) {
        // Empty array
        throw std::runtime_error("Dot product of empty array");
    } else if (lhs.size() == 1) {
        // Simple multiply (left)
        return static_cast<float>(lhs) * rhs;
    } else if (rhs.size() == 1) {
        // Simple multiply (right)
        return lhs * static_cast<float>(rhs);
    } else if (l_shape.size() == 1 && r_shape.size() == 1) {
        // Inner product of vector (1D, 1D)
        return DotTensor1d(lhs, rhs);
    } else if (r_shape.size() == 1) {
        // Broadcast right 1D array
        const Shape shape(l_shape.begin(), l_shape.end() - 1);
        return DotTensorNdMd(lhs, rhs.reshape(r_shape[0], 1)).reshape(shape);
    } else {
        // Basic matrix product
        return DotTensorNdMd(lhs, rhs);
    }
}

Tensor Tensor::dot(Tensor& other){
    //return DotTensor(*this, other);
    Tensor ret = DotTensor(*this, other);
    if(this->requires_grad() || other.requires_grad()){
        ret.requires_grad(true);
        ret.has_ctx = true;
        ret.ctx = std::make_shared<dot_op>(this, &other);
    }
    return ret;
}

// --------------------------- Print tensor --------------------------------


static void OutputArrayLine(std::ostream& os, const float* data,
                            const int size) {
    os << "[";  // Begin of a line
    for (int i = 0; i < size; i++) {
        os << data[i];  // Output an element
        if (i == size - 1) {
            os << "]";  // End of a line
        } else {
            os << ", ";  // Splitter of an element
        }
    }
}

static void OutputArrayMultiDim(std::ostream& os,
                                const float* data,
                                const Shape& shape,
                                const std::vector<int>& child_sizes,
                                size_t depth) {
    for (int i = 0; i < shape[depth]; i++) {
        // Heading
        if (i == 0) {
            os << "[";  // begin of array
        } else {
            for (size_t d = 0; d < depth + 1; d++) {  // array indent
                os << " ";
            }
        }

        // Output internal array
        const int& child_size = child_sizes[depth];
        if (depth == shape.size() - 2) {
            OutputArrayLine(os, data + child_size * i, shape[depth + 1]);
        } else {
            OutputArrayMultiDim(os, data + child_size * i, shape, child_sizes,
                                depth + 1);
        }

        // Tailing
        if (i == shape[depth] - 1) {
            os << "]";  // End of array
        } else {
            os << "," << std::endl;  // Splitter of array
        }
    }
}

static void OutputTensor(std::ostream& os, const Tensor& x) {
    const int size = static_cast<int>(x.size());
    const Shape& shape = x.shape();
    const std::vector<int>& child_sizes = ComputeChildSizes(shape);

    if (size == 0 || shape.size() == 0) {
        // Empty
        os << "[]";
    } else if (shape.size() == 1) {
        // 1-dim
        OutputArrayLine(os, x.data(), size);
    } else {
        // Multi-dim
        OutputArrayMultiDim(os, x.data(), shape, child_sizes, 0);
    }
}

static void OutputShape(std::ostream& os, const Shape& shape) {
    os << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        os << shape[i];
        if (i < shape.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
}

std::ostream& operator<<(std::ostream& os, const Tensor& x) {
    OutputTensor(os, x);
    return os;
}

std::ostream& operator<<(std::ostream& os, const Shape& s){
    OutputShape(os, s);
    return os;
}

// -------------- Utilities for NdArray (Dual operator in-place) ---------------

template <typename F>
Tensor ApplyDualOpInplace(Tensor&& lhs, Tensor&& rhs, F op,
                           const bool allow_new = true) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        ApplyOpSimple(lhs, lhs, rhs, op);  // Use left as result
        return std::move(lhs);
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Select result with shape matching
        Tensor&& ret =
                (ret_shape == lhs.shape()) ? lhs :                  // left
                        (ret_shape == rhs.shape()) ? rhs :          // right
                                (allow_new) ? Tensor(ret_shape) :  // new
                                        throw std::runtime_error(
                                                "Invalid shape for in-place"
                                                " operation");
        // Apply broadcast
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(op));
        return std::move(ret);
    }
}

template <typename F>
Tensor ApplyDualOpInplace(Tensor&& lhs, const Tensor& rhs, F op,
                           const bool allow_new = true) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        ApplyOpSimple(lhs, lhs, rhs, op);
        return std::move(lhs);
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Select result with shape matching
        Tensor&& ret =
                (ret_shape == lhs.shape()) ? lhs :          // left
                        (allow_new) ? Tensor(ret_shape) :  // new
                                throw std::runtime_error(
                                        "Invalid shape for in-place operation");
        // Apply broadcast (result matrix is lhs)
        ApplyOpBroadcast(ret, lhs, rhs, 0, 1, WrapOpForIter(op));
        return std::move(ret);
    }
}

template <typename F>
inline Tensor ApplyDualOpInplace(const Tensor& lhs, Tensor&& rhs, F op,
                                  const bool allow_new = true) {
    // Swap left and right
    return ApplyDualOpInplace(std::move(rhs), lhs, ReverseOp(op), allow_new);
}

template <typename F>
Tensor ApplyDualOpInplace(Tensor&& lhs, float rhs, F op) {
    // Broadcast right float
    // Simply apply all
    ApplyOpSimple(lhs, lhs, rhs, op);
    return std::move(lhs);
}

template <typename F>
inline Tensor ApplyDualOpInplace(float lhs, Tensor&& rhs, F op) {
    // Swap left and right
    return ApplyDualOpInplace(std::move(rhs), lhs, ReverseOp(op));
}

// ------------------------------- Transpose -----------------------------------

template <typename ViewF>
static void ChangeTensorView(float* ret_data,
                              const float* src_data, size_t size,
                              ViewF view_func) {
    // Copy under view conversion function
    for(int src_idx = 0; src_idx < size; src_idx++){
        const int ret_idx = view_func(src_idx);
        ret_data[ret_idx] = src_data[src_idx];
    }
}


static Tensor TransposeTensor(const Tensor& src) {
    const Shape& src_shape = src.shape();

    // Create result shape
    Shape ret_shape;
    std::reverse_copy(src_shape.begin(), src_shape.end(),
                      back_inserter(ret_shape));
    // Create result array
    Tensor ret(ret_shape);

    // Pre-compute child sizes
    const std::vector<int> src_child_sizes = ComputeChildSizes(src_shape);
    const std::vector<int> ret_child_sizes = ComputeChildSizes(ret_shape);

    // Apply view change
    const size_t ndim = src_shape.size();
    ChangeTensorView(ret.data(), src.data(), src.size(), [&](int src_idx) {
        // Decompose
        auto src_idxs = std::make_unique<int[]>(ndim);
        for (size_t d = 0; d < ndim; d++) {
            src_idxs[d] = src_idx / src_child_sizes[d];
            src_idx -= src_idxs[d] * src_child_sizes[d];
        }
        // Compose (with inverting indices)
        int ret_idx = 0;
        for (size_t d = 0; d < ndim; d++) {
            ret_idx += src_idxs[ndim - d - 1] * ret_child_sizes[d];
        }
        return ret_idx;
    });
    return ret;
}

static Tensor BroadcastToTensor(const Tensor& x, const Shape& shape) {
    Tensor ret(shape);
    return ApplyDualOpInplace(
            std::move(ret), x, [](float, float r) { return r; }, false);
}

Tensor Transpose(const Tensor& x) {
    return TransposeTensor(x);
}

// ------------------------------- Reshape Method ------------------------------
Tensor Tensor::reshape(const Shape& shape) const {
    // Check shape validity
    size_t unknown_idx = shape.size();
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] < 0) {
            if (unknown_idx != shape.size()) {
                throw std::runtime_error("Invalid shape format (multi-neg)");
            } else {
                unknown_idx = i;
            }
        } else {
            size *= static_cast<size_t>(shape[i]);
        }
    }
    Shape new_shape = shape;
    if (unknown_idx == shape.size()) {
        if (values->size != size) {
            std::stringstream ss;
            ss << "Invalid reshape (" << values->size << "->" << size << ")";
            throw std::runtime_error(ss.str());
        }
    } else {
        if (values->size % size != 0) {
            throw std::runtime_error("Invalid reshape (-1)");
        }
        new_shape[unknown_idx] = static_cast<int>(values->size / size);
    }

    // Create reshaped array
    Tensor ret;
    ret.values->size = values->size;           // Same size
    ret.values->shape = std::move(new_shape);  // New shape
    ret.values->vals = values->vals;           // Shared elements
    return ret;
}

template <typename... S>
Tensor Tensor::reshape(S... shape) const {
    return reshape({shape...});
}

Tensor Tensor::copy() const {
    auto sub = std::make_shared<Substance>(values->size, values->shape);
    Tensor ret(sub);
    ApplyOpSimple(ret, *this, [](const float& x) {return x;});
    return ret;
}

Tensor Tensor::flatten() const{
    return reshape({-1}).copy();
}

Tensor Tensor::ravel() const {
    return reshape({-1});
}

template Tensor Tensor::reshape(int) const;
template Tensor Tensor::reshape(int, int) const;
template Tensor Tensor::reshape(int, int, int) const;
template Tensor Tensor::reshape(int, int, int, int) const;
template Tensor Tensor::reshape(int, int, int, int, int) const;
template Tensor Tensor::reshape(int, int, int, int, int, int) const;
template Tensor Tensor::reshape(int, int, int, int, int, int, int) const;
template Tensor Tensor::reshape(int, int, int, int, int, int, int, int) const;
template Tensor Tensor::reshape(int, int, int, int, int, int, int, int,
                                  int) const;
template Tensor Tensor::reshape(int, int, int, int, int, int, int, int, int,
                                  int) const;
template Tensor Tensor::reshape(int, int, int, int, int, int, int, int, int,
                                  int, int) const;

// For `Ones(S... shape)`
template Tensor Tensor::Ones(int);
template Tensor Tensor::Ones(int, int);
template Tensor Tensor::Ones(int, int, int);
template Tensor Tensor::Ones(int, int, int, int);
template Tensor Tensor::Ones(int, int, int, int, int);
template Tensor Tensor::Ones(int, int, int, int, int, int);
template Tensor Tensor::Ones(int, int, int, int, int, int, int);
template Tensor Tensor::Ones(int, int, int, int, int, int, int, int);
template Tensor Tensor::Ones(int, int, int, int, int, int, int, int, int);
template Tensor Tensor::Ones(int, int, int, int, int, int, int, int, int,
                               int);


// ----------------------- Backwards -----------------------------


/**
 * Topological sort of the operations tree
 * Recursive walk
*/
void _deepwalk(Tensor* node, std::set<Tensor* >& visited, std::vector<Tensor*>& nodes){
    const bool is_in = visited.find(node) != visited.end();
    if (!is_in){
        visited.insert(node);
        if(node->has_ctx == 1){
            for(Tensor* n : node->ctx->parents){
                _deepwalk(n, visited, nodes);
            }
        }
        nodes.push_back(node);
    }
}

struct Graph{
    std::set<Tensor*>visited;
    std::vector<Tensor*>nodes;
};

/**
 * Topological sort of the operations tree
*/
Graph Tensor::deepwalk(){
    std::set<Tensor*> visited;
    std::vector<Tensor*> nodes;
    _deepwalk(this, visited, nodes);

    return Graph{
        visited = visited,
        nodes = nodes
    };
}

/**
 * The backwards method for a tensor
 * When called -> 
 *      built ops graph
 *      compute gradients for each parent
*/
void Tensor::backward(){
    //* Initialise the gradient as 1
    this->grad = std::make_shared<Tensor>(this->values->shape, 1.f);

    //* Collect the operational graph
    Graph g = deepwalk();

    //* Reverse the graph
    std::reverse(g.nodes.begin(), g.nodes.end());

    //for(auto& n : g.nodes){
    for(Tensor* n: g.nodes){
        if( n->has_ctx == true ){
            n->ctx->backward(n);
        }
    }
}