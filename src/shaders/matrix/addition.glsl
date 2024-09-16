#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) buffer MatrixA { float matrixA[]; };
layout (binding = 1) buffer MatrixC { float matrixC[]; };
layout (binding = 2) buffer MatrixB { float matrixB[]; };

uniform int colsA;
uniform int rowsA;
uniform int colsB;
uniform int rowsB;

void main() {
    int index = int(gl_GlobalInvocationID.x);
    matrixC[index] = matrixA[index] + matrixB[index];
}