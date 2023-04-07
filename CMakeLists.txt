cmake_minimum_required(VERSION 3.25)
project(CustomNN)

set(CMAKE_CXX_STANDARD 17)


include_directories(${./include})
add_executable(CustomNN main.cpp src/matrix.cpp include/nueral_network.h src/nueral_network.cpp)
