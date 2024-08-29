#version 430

layout(local_size_x = 1) in;

layout(binding = 0) buffer inputBuffer1 { float carData[]; };
layout(binding = 1) buffer inputBuffer2 { float insidePoints[]; };
layout(binding = 2) buffer inputBuffer3 { float outsidePoints[]; };
layout(binding = 3) buffer inputBuffer4 {float visionAngles[]; };
layout(binding = 4) buffer outputBuffer {float distances[]; };

uniform int numCarFloats;
uniform int numInsidePoints;
uniform int numOutsidePoints;
uniform int numComputerVisionAngles;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    uint inIndex = index * numCarFloats;
    uint outIndex = index * numComputerVisionAngles;

    for (int i = 0; i < numComputerVisionAngles; i++) {
        distances[outIndex + i] = 0.05;
    }
}

