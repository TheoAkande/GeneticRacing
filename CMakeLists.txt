cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenGL REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Add the library directory
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Add source files
add_executable(MyProject 
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/Utils.cpp
)

# Link the libraries
target_link_libraries(MyProject glew32 glfw3 soil2-debug opengl32)

# Copy GLEW DLL to the output directory
add_custom_command(TARGET MyProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PROJECT_SOURCE_DIR}/glew32.dll"
    $<TARGET_FILE_DIR:MyProject>)

file(COPY ${CMAKE_SOURCE_DIR}/src/shaders DESTINATION ${CMAKE_BINARY_DIR})

add_custom_command(
    TARGET MyProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/src/shaders $<TARGET_FILE_DIR:MyProject>/shaders
)

file(COPY ${CMAKE_SOURCE_DIR}/src/assets DESTINATION ${CMAKE_BINARY_DIR})

# Post-build event to copy shaders again after each build
add_custom_command(
    TARGET MyProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/src/assets $<TARGET_FILE_DIR:MyProject>/assets
)