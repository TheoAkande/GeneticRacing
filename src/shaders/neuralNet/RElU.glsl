#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) buffer MatrixA { float matrixA[]; };

uniform int rows;
uniform int cols;

void main() {
    uint index = gl_GlobalInvocationID.x;
    matrixA[index] = max(0.0, matrixA[index]);
}