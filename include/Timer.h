//
// Created by alan on 18/07/23.
//

#ifndef CUSTOMNN_TIMER_H
#define CUSTOMNN_TIMER_H

#include <chrono>

class Timer {
public:
    void start();
    void stop();
    double getCurrentTimeInSeconds();

private:
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::steady_clock::duration> _startTime;
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::steady_clock::duration> _stopTime;
};



#endif //CUSTOMNN_TIMER_H
