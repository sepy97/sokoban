cmake_minimum_required(VERSION 3.13)  # CMake version check
project(sokoban)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 17)            # Enable c++14 standard

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES main.cpp ./sokoban/sokoban.cpp ./sokoban/sokoban.h)

configure_file(sokoban00.txt ${CMAKE_CURRENT_BINARY_DIR}/sokoban00.txt COPYONLY)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(sokoban ${SOURCE_FILES})
