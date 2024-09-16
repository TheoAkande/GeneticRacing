#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) buffer MatrixA { float matrixA[]; };
layout (binding = 1) buffer MatrixB { float matrixB[]; };

uniform int cols;
uniform int rows;

uniform float scalar;

void main() {
    uint index = gl_GlobalInvocationID.x;
    matrixB[index] = matrixA[index] * scalar;
}