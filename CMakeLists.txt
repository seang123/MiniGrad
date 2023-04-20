cmake_minimum_required(VERSION 3.10.2)
project(MiniGrad)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -static -pthread -msse4.1")

set(SOURCE_FILES main.cpp Tensor.cpp Substance.cpp Ops.cpp, Operations.cpp nn.cpp)


if(UNIX)
    message("This is a ${CMAKE_SYSTEM_NAME} system")
elseif(WIN32)
    message("This is a Windows System")
endif()

#add_executable(cmake_MiniGrad $(SOURCE_FILES))
add_executable(a.exe main.cpp Tensor.cpp Substance.cpp Ops.cpp Operations.cpp nn.cpp )