#include <iostream>
#include <vector>
#include <memory>

#include "Iter.h"


// Constructor
Iter::Iter(float* p_) : p(p_) {};

// Desctructor
Iter::~Iter() {};

float& Iter::operator*() const {
    return *p;
}

float& Iter::operator[](int i) const {
    return p[i];
}

Iter& Iter::operator++(){
    p++;
    return *this;
}

Iter& Iter::operator--(){
    p--;
    return *this;
}


Iter Iter::operator++(int){
    Iter tmp = *this;
    p++;
    return tmp;
}

Iter Iter::operator--(int){
    Iter tmp = *this;
    p--;
    return tmp;
}

// ??
Iter Iter::operator+(int i) const {
    return {p+i};
}

// ??
Iter Iter::operator-(int i) const {
    return {p-i};
}

Iter& Iter::operator+=(int i) {
    p += i;
    return *this;
}

Iter& Iter::operator-=(int i) {
    p -= i;
    return *this;
}

bool Iter::operator==(const Iter& other) const {
    return p == other.p;
}

bool Iter::operator!=(const Iter& other) const {
    return p != other.p;
}
