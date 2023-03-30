
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
        CHECK(tensor_to_str(t3) == "[[4, 5, 6]]\n");
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

        CHECK(tensor_to_str(*t1.grad) == "[[1, 1, 1]]\n");
        CHECK(tensor_to_str(*t2.grad) == "[[1, 1, 1]]\n");
    }

    SUBCASE("multiplication gradients"){
        Tensor t6 = t1 * t2;

        CHECK(tensor_to_str(t6) == "[[3, 6, 9]]\n");

        t6.backward();
        CHECK(tensor_to_str(*t1.grad) == "[[3, 3, 3]]\n");
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

        CHECK(tensor_to_str(*a.grad) == "[6]\n");
        CHECK(tensor_to_str(*b.grad) == "[-4]\n");
        CHECK(tensor_to_str(*c.grad) == "[-2]\n");
        CHECK(tensor_to_str(*e.grad) == "[-2]\n");
        CHECK(tensor_to_str(*d.grad) == "[-2]\n");
        CHECK(tensor_to_str(*f.grad) == "[4]\n");
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

        std::cout << "e: " << e << "\n";
        std::cout << "d: " << d << "\n";
        std::cout << "L: " << L << "\n";
        
        L.backward();

        std::cout << *f.grad << "\n";
    }

}