#ifndef _Iter_H_
#define _Iter_H_

#include <iostream>
#include <vector>
#include <memory>
#include <cstddef>

/*
Acts as a pointer to the data array on the heap
Instead of passing around a raw pointer, we pass around a reference to this class
*/

class Iter{

private:
    float* p;

public:
    // Below setting make Iter a standard library compatible iterator
    using iterator_category = std::random_access_iterator_tag;
    using value_type = float; // the type of an instance of this class is equivalent to float
    using difference_type = std::ptrdiff_t; // allows pointer arithmatic
    using pointer = value_type*;
    using reference = value_type&;

    Iter(float* p_);
    ~Iter();

    float& operator*() const;
    float& operator[](int i) const;
    Iter& operator++();
    Iter& operator--();
    Iter operator++(int);
    Iter operator--(int);
    Iter operator+(int i) const;
    Iter operator-(int i) const;
    Iter& operator+=(int i);
    Iter& operator-=(int i);
    bool operator==(const Iter& other) const;
    bool operator!=(const Iter& other) const;
};

#endif