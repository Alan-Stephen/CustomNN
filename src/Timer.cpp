//
// Created by alan on 18/07/23.
//

#include "../include/Timer.h"

void Timer::start() {
    _startTime = std::chrono::steady_clock::now();
}

void Timer::stop() {
    _stopTime = std::chrono::steady_clock::now();
}

double Timer::getCurrentTimeInSeconds() {
    const std::chrono::duration<double> elapsedTime = _stopTime - _startTime;

    return elapsedTime.count();
}