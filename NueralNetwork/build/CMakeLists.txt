cmake_minimum_required(VERSION 3.12)
project(NueralNetwork VERSION 1.0) 

file(GLOB_RECURSE SRC_FILES ../src/*.cpp)
add_executable(NueralNetwork ${SRC_FILES})  
target_include_directories(NueralNetwork PUBLIC ../include)

