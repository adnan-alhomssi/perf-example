cmake_minimum_required(VERSION 3.13)
project(perf_example)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -mavx2")

add_executable(perf_example main.cpp)