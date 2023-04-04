
#include <iostream>
#include <vector>

#include "Tensor.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Ops.h"
#include "Operations.h"

//using namespace Ops;
using std::cout;

static std::string tensor_to_str(const Tensor& t){
    std::stringstream ss;
    ss << t;
    return ss.str();
}

static std::string shape_to_str(const Shape& s){
    std::stringstream ss;
    ss << s;
    return ss.str();
}

// ------------------- test print function -----------------

std::string print_tensor(){
    Tensor t = {{1.f, 2.f, 3.f}};
    std::stringstream ss;
    ss << t;
    return ss.str();
}

std::string print_tensor_2d(){
    Tensor t = {{{1.1f, 2.2f}, {3.3f, 4.4f}}};
    std::stringstream ss;
    ss << t;
    return ss.str();
}

std::string print_tensor_shape(){
    Tensor t = {{1.1f, 2.2f}, {3.3f, 4.4f}}; // 2x2
    std::stringstream ss;
    ss << t.shape();
    return ss.str();
}

TEST_CASE("tensor print"){
    std::cout << "-- tensor printing\n";
    CHECK(print_tensor() == "[[1, 2, 3]]");
    CHECK(print_tensor_2d() == "[[[1.1, 2.2],\n  [3.3, 4.4]]]");
    CHECK(print_tensor_shape() == "[2, 2]");
}

TEST_CASE("Init"){

    SUBCASE("Ones"){
        Tensor t1 = Tensor::Ones(2, 3);
        CHECK(shape_to_str(t1.shape()) == "[2, 3]");
    }

    SUBCASE("Zeros"){

    }

    SUBCASE("Shape"){
        Tensor t1 (Shape{3, 2, 4});
        CHECK(shape_to_str(t1.shape()) == "[3, 2, 4]");
    }

    SUBCASE("Shape + fill"){
        Tensor t1 (Shape{2, 2}, 4.0f);
        CHECK(shape_to_str(t1.shape()) == "[2, 2]");
        CHECK(tensor_to_str(t1) == "[[4, 4],\n [4, 4]]");
    }
}

TEST_CASE("basic operations"){
    std::cout << "-- basic operations\n";
    Tensor t1 = {-1.f, 1.f, 2.f, 3.f};

    SUBCASE("Empty") {
        const Tensor m1;
        CHECK(m1.empty());
        CHECK(m1.size() == 0);
        CHECK(m1.shape() == Shape{0});
        CHECK(m1.ndim() == 1);
    }
    SUBCASE("Reshape"){
        //Tensor t1 = {{1.f, 2.f, 3.f}};
        t1 = t1.reshape(4, 1);
        //CHECK(tensor_to_str(t1) == "[[1],\n [2],\n [3]]");
        CHECK(tensor_to_str(t1) == "[[-1],\n [1],\n [2],\n [3]]");
    }

    SUBCASE("Minus scalar"){
        Tensor t2 = 1.f - t1;
        CHECK(tensor_to_str(t2) == "[2, 0, -1, -2]");

        Tensor t3 = t1 - 1.f;
        CHECK(tensor_to_str(t3) == "[-2, 0, 1, 2]");
    }

    SUBCASE("Add scalar"){
        Tensor t2 = 1.f + t1;
        CHECK(tensor_to_str(t2) == "[0, 2, 3, 4]");

        Tensor t3 = t1 + 1.f;
        CHECK(tensor_to_str(t3) == "[0, 2, 3, 4]");
    }
}

TEST_CASE("gradient accumulation"){
    std::cout << "-- gradient accumulation\n";
    Tensor t1 = {{1.f, 2.f, 3.f}}; // (1, 3)
    Tensor t2 = {3.f, 3.f, 3.f}; // (3,)
    t1.requires_grad(true);
    t2.requires_grad(true);
    t1.name = "t1";
    t2.name = "t2";

    Tensor t4 = {{4.f, 44.f, 444.f}};
    t4.requires_grad(true);
    t4.name = "t4";

    CHECK(t1.requires_grad() == true);
    CHECK(t2.requires_grad() == true);
    CHECK(t4.requires_grad() == true);

    SUBCASE("addition gradients"){
        Tensor t3 = t1 + t2;
        CHECK(tensor_to_str(t3) == "[[4, 5, 6]]");
        CHECK(t3.requires_grad() == true); // if input requires_grad -> output requires_grad
        CHECK(t3.has_ctx == 1);
        t3.name = "t3";


        CHECK(t4.requires_grad() == true);
        CHECK(t4.has_ctx == false);

        Tensor t5 = t4 + t3;
        t5.name = "t5";
        t5.backward();

        CHECK(t5.requires_grad() == true);
        CHECK(t5.has_ctx == true);

        CHECK(tensor_to_str(*t1.grad) == "[[1, 1, 1]]");
        CHECK(tensor_to_str(*t2.grad) == "[[1, 1, 1]]");
    }

    SUBCASE("multiplication gradients"){
        Tensor t6 = t1 * t2;

        CHECK(tensor_to_str(t6) == "[[3, 6, 9]]");

        t6.backward();
        CHECK(tensor_to_str(*t1.grad) == "[[3, 3, 3]]");
    }

    SUBCASE("karpathy example"){

        Tensor a ({2.f});
        a.requires_grad(true);
        Tensor b ({-3.f});
        b.requires_grad(true);
        Tensor c ({10.f});
        c.requires_grad(true);
        Tensor e = a * b; // 2 * -3 = -6
        Tensor d = e + c; // -6 + 10 = 4
        Tensor f ({-2.f});
        f.requires_grad(true);
        Tensor L = d * f; // 4 * -2 = -8

        //* Gradients will be as follows:
        //? dL / dL = 1
        //? dL / dd = f = -2
        //? dL / df = d = 4
        //? dL / dc = dL/dd * dd/dc => -2 * 1 = -2
        //? dL / de = dL/dd * dd/de => -2 * 1 = -2
        //? dL / db = dd/de * de/db = -2 * 2 = -4
        //? dL / da = dL/de * de/da => -2 * b = -2 * -3 = 6

        // dL/dd = (f(x + h) - f(x)) / h
        //       = ((d+h)*f - d*f) / h
        //       = (d*f + h*f - d*f) / h
        //       = (h*f) / h
        //       = f

        // dd/dc = (f(x+h) - f(x)) / h
        //       = ((c+h + e) - (c+e)) / h
        //       = ( c + h + e - c - e) / h
        //       = ( h ) / h
        //       = 1.0
        // dd/de = 1.0 by symmetry

        L.backward();

        CHECK(tensor_to_str(*a.grad) == "[6]");
        CHECK(tensor_to_str(*b.grad) == "[-4]");
        CHECK(tensor_to_str(*c.grad) == "[-2]");
        CHECK(tensor_to_str(*e.grad) == "[-2]");
        CHECK(tensor_to_str(*d.grad) == "[-2]");
        CHECK(tensor_to_str(*f.grad) == "[4]");
    }

    SUBCASE("vector example"){
        Tensor a ({3.f, 4.f, 2.f, 4.f, 4.f});
        a.requires_grad(true);
        Tensor b({1.f, 2.f, 2.f, 2.f, 4.f});
        b.requires_grad(true);
        Tensor c({3.f, 2.f, 4.f, 1.f, 3.f});
        c.requires_grad(true);
        Tensor e = a * b;
        Tensor d = e + c;
        Tensor f({-2.f, -2.f, -2.f, -2.f, -2.f});
        f.requires_grad(true);
        Tensor L = d * f;
        
        L.backward();

        CHECK(tensor_to_str(*a.grad) == "[-2, -4, -4, -4, -8]");
        CHECK(tensor_to_str(*b.grad) == "[-6, -8, -4, -8, -8]");
        CHECK(tensor_to_str(*c.grad) == "[-2, -2, -2, -2, -2]");
        CHECK(tensor_to_str(*e.grad) == "[-2, -2, -2, -2, -2]");
        CHECK(tensor_to_str(*d.grad) == "[-2, -2, -2, -2, -2]");
        CHECK(tensor_to_str(*f.grad) == "[6, 10, 8, 9, 19]");
    }
}

TEST_CASE("Operations"){

    Tensor t1 ({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f});

    SUBCASE("tanh"){
        t1.requires_grad(true);

        Tensor out = Ops::tanh(t1);
        out.backward();

        //* Assumes normal math.h exp not SIMD method for tanh computation
        CHECK(tensor_to_str(out) == "[-0.761594, 0, 0.761594, 0.964028, 0.995055, 0.999329, 0.999909]");
        CHECK(tensor_to_str(*t1.grad) == "[0.419974, 1, 0.419974, 0.0706508, 0.009866, 0.00134087, 0.000181556]");
    }

    SUBCASE("exponential "){
        t1.requires_grad(true);
        Tensor out = Ops::exp(t1);
        CHECK(tensor_to_str(out) == "[0.367879, 1, 2.71828, 7.38906, 20.0855, 54.5981, 148.413]");

        out.backward();
        // y = e^x  ->  dy/dx = e^x
        CHECK(tensor_to_str(out) == tensor_to_str(*t1.grad));
    }

    SUBCASE("Tensor::square()"){
        Tensor out = t1.square();
        CHECK(tensor_to_str(out) == "[1, 0, 1, 4, 9, 16, 25]");
    }


    SUBCASE("Dot product"){
        Tensor::SetNumWorkers(2);
        Tensor t2 = Tensor::Uniform(Shape{32, 3, 16});
        Tensor t3 = Tensor::Uniform(Shape{16, 2});

        Tensor t4 = t2.dot(t3);

        cout << "t4.shape(): " << t4.shape() << "\n";
    }

}

TEST_CASE("Random"){
    SUBCASE("Uniform"){
        Tensor t1 = Tensor::Uniform(Shape{2, 3});
    }

    SUBCASE("seed"){
        Tensor::Seed(42);
        Tensor t1 = Tensor::Uniform(Shape{2, 2});

        Tensor::Seed(42);
        Tensor t2 = Tensor::Uniform(Shape{2, 2});

        CHECK(tensor_to_str(t1) == tensor_to_str(t2));
    }

}