#version 430

layout(local_size_x = 1) in;

layout (binding = 0) buffer inputBuffer1 { float carData[]; };
layout (binding = 1) buffer inputBuffer2 { int carLaps[]; };
layout (binding = 2) buffer outputBuffer {float fitness[]; };

uniform int numCarFloats;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    uint inIndex = index * numCarFloats;
    
    fitness[index] = carData[inIndex + 8] + (float) carLaps[index] * 50.0;
}