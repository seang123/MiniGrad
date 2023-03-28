
#include <iostream>
#include <vector>

#include "Tensor.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Ops.h"


//using std::cout;


std::string tensor_to_str(const Tensor& t){
    std::stringstream ss;
    ss << t << "\n";
    return ss.str();
}

// ------------------- test print function -----------------

std::string print_tensor(){
    Tensor t = {{1.f, 2.f, 3.f}};
    std::stringstream ss;
    ss << t << "\n";
    return ss.str();
}

std::string print_tensor_2d(){
    Tensor t = {{{1.1f, 2.2f}, {3.3f, 4.4f}}};
    std::stringstream ss;
    ss << t << "\n";
    return ss.str();
}

std::string print_tensor_shape(){
    Tensor t = {{1.1f, 2.2f}, {3.3f, 4.4f}}; // 2x2
    std::stringstream ss;
    ss << t.shape() << "\n";
    return ss.str();
}

// ------------------- Addition ---------------

std::string tensor_addition(){
    const Tensor t1 = {{1.f, 2.f, 3.f}};
    const Tensor t2 = {{4.f, 5.f, 6.f}};
    Tensor t3 = t1 + t2;
    std::stringstream ss;
    ss << t3 << "\n";
    return ss.str();
}

std::string tensor_addition_broadcast(){
    const Tensor t1 = {{1.f, 2.f}, {3.f, 4.f}};  // 2x2
    const Tensor t2 = {{4.f, 5.f}};  // 1x2
    const Tensor t3 = t1 + t2;
    std::stringstream ss;
    ss << t3 << "\n";
    return ss.str();
}

// --------------- basic ops --------------------

/*Tensor equality(){
    Tensor t = {{1, 2, 3}};
    return t;
}*/

TEST_CASE("tensor print"){
    std::cout << "-- tensor printing\n";
    CHECK(print_tensor() == "[[1, 2, 3]]\n");
    CHECK(print_tensor_2d() == "[[[1.1, 2.2],\n  [3.3, 4.4]]]\n");
    CHECK(print_tensor_shape() == "[2, 2]\n");
}

TEST_CASE("tensor addition"){
    std::cout << "-- tenor addition\n";
    CHECK(tensor_addition() == "[[5, 7, 9]]\n");
    CHECK(tensor_addition_broadcast() == "[[5, 7],\n [7, 9]]\n");
}

TEST_CASE("basic operations"){
    std::cout << "-- basic operations\n";
    SUBCASE("Empty") {
        const Tensor m1;
        CHECK(m1.empty());
        CHECK(m1.size() == 0);
        CHECK(m1.shape() == Shape{0});
        CHECK(m1.ndim() == 1);
    }
    SUBCASE("Reshape"){
        Tensor t1 = {{1.f, 2.f, 3.f}};
        t1 = t1.reshape(3, 1);
        CHECK(tensor_to_str(t1) == "[[1],\n [2],\n [3]]\n");
    }
    //CHECK(equality() == Tensor({{1, 2, 3}}));
}

TEST_CASE("gradient accumulation"){
    std::cout << "-- gradient accumulation\n";
    Tensor t1 = {{1.f, 2.f, 3.f}}; // (1, 3)
    Tensor t2 = {3.f, 3.f, 3.f}; // (3,)
    t1.requires_grad(true);
    t2.requires_grad(true);

    CHECK(t1.requires_grad() == true);
    CHECK(t2.requires_grad() == true);

    Tensor t3 = t1 + t2;
    CHECK(tensor_to_str(t3) == "[[4, 5, 6]]\n");
    CHECK(t3.requires_grad() == true); // if input requires_grad -> output requires_grad
    CHECK(t3.has_ctx == 1);

    // backwards pass
    t3.backward();

    CHECK(t3.ctx->parents.size() == 2);

    std::cout << *t1.grad << "\n";
    std::cout << *t2.grad << "\n";
}