cmake_minimum_required(VERSION 3.25.2)
project(MiniGrad)
set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp Tensor.cpp Substance.cpp Iter.cpp)

add_executable(MiniGrad $(SOURCE_FILES))