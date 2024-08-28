#version 430

layout(local_size_x = 1) in;

layout(binding = 0) buffer inputBuffer { float carData[]; };
layout(binding = 1) buffer outputBuffer {float carPoints[]; };

uniform int numCarFloats;
uniform float carWidth;
uniform float carHeight;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    uint inIndex = index * numCarFloats;
    
    float x = carData[inIndex];
    float y = carData[inIndex + 1];
    vec2 xy = vec2(x, y);

    vec2 frontRight = vec2(carHeight * 2.0 / 3.0, carWidth / 2.0);
    vec2 frontLeft = vec2(carHeight * 2.0 / 3.0, -1.0 * carWidth / 2.0);
    vec2 backRight = vec2(-1.0 * carHeight / 3.0, carWidth / 2.0);
    vec2 backLeft = vec2(-1.0 * carHeight / 3.0, -1.0 * carWidth / 2.0);

    float angle = -carData[inIndex + 2];

    mat2 rotation = mat2(
        cos(angle), -sin(angle), 
        sin(angle), cos(angle)
    );

    frontRight = rotation * frontRight + xy;
    frontLeft = rotation * frontLeft + xy;
    backRight = rotation * backRight + xy;
    backLeft = rotation * backLeft + xy;

    uint outIndex = index * 10;
    carPoints[outIndex] = frontRight.x;
    carPoints[outIndex + 1] = frontRight.y;
    carPoints[outIndex + 2] = frontLeft.x;
    carPoints[outIndex + 3] = frontLeft.y;
    carPoints[outIndex + 4] = backRight.x;
    carPoints[outIndex + 5] = backRight.y;
    carPoints[outIndex + 6] = backLeft.x;
    carPoints[outIndex + 7] = backLeft.y;
    carPoints[outIndex + 8] = x;
    carPoints[outIndex + 9] = y;
}