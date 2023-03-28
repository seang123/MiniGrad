#include <iostream>
#include <vector>

#include "Tensor.h"
#include "Iter.h"

//#define TINYNDARRAY_IMPLEMENTATION
//#include "tinyndarray.h"
//using tinyndarray::NdArray;

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


int main()
{
    basic_test();

    return 0;
}
