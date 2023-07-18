cmake_minimum_required(VERSION 3.25)
project(CustomNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${./include})
add_executable(CustomNN main.cpp include/matrix.h src/matrix.cpp include/LinearLayer.h
        include/LinearLayer.h src/LinearLayer.cpp include/py_net.h src/py_net.cpp include/SigmoidLayer.h include/Layer.h include/MNISTLabel.h src/MNISTLabel.cpp src/SigmoidLayer.cpp include/NetworkBlueprint.h src/LinearLayer.cpp include/ReLuLayer.h src/ReLuLayer.cpp include/SoftmaxLayer.h src/SoftmaxLayer.cpp include/Conv2DLayer.h src/Conv2DLayer.cpp include/Timer.h src/Timer.cpp)
