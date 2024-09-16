#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) buffer MatrixA { float matrixA[]; };
layout (binding = 1) buffer MatrixB { float matrixB[]; };

uniform int colsA;
uniform int rowsA;

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint row = index / colsA;
    uint col = index % colsA;
    uint transposedIndex = col * rowsA + row;
    matrixB[transposedIndex] = matrixA[index];
}