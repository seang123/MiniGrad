#ifndef _Substance_H_
#define _Substance_H_

#include <iostream>
#include <vector>
#include <memory>

using Shape = std::vector<int>;

/*
Stores a pointer to the raw data array - created dynamically on the heap.
Also stores the shape of the ndarray
and the size of the array - product of shape dimensions
*/
class Substance{
public:
    Substance(size_t size_ = 0, const Shape& shape = {0}) 
        : shape(shape),
          size(size_),
          vals(new float[size_], std::default_delete<float[]>()){};
    ~Substance();

    Shape shape;
    size_t size;
    alignas(32) std::shared_ptr<float> vals; // the raw data array - on the heap


    float* begin(){
        return vals.get();
    }

    float* end(){
        return vals.get() + size-1;
    }

};

#endif
