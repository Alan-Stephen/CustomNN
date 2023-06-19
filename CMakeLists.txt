cmake_minimum_required(VERSION 3.25)
project(CustomNN)

set(CMAKE_CXX_STANDARD 17)


include_directories(${./include})
add_executable(CustomNN main.cpp include/matrix.h src/matrix.cpp include/LinearLayer.h
        include/LinearLayer.h src/LinearLayer.cpp include/py_net.h src/py_net.cpp include/SigmoidLayer.h include/Layer.h include/MNISTLabel.h src/MNISTLabel.cpp)
