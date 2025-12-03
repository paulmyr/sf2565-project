cmake_minimum_required(VERSION 3.10)
project(sf2565_project)

set(CMAKE_CXX_STANDARD 20)

# Add Eigen from submodule
add_subdirectory(lib/eigen)
add_executable(sf2565_project src/main.cpp
        src/arnoldi.h
        src/lanczos.cpp
        src/lanczos.h
)

# CLI11 header
target_include_directories(sf2565_project PRIVATE lib/cli11)
target_link_libraries(sf2565_project
    Eigen3::Eigen
)
