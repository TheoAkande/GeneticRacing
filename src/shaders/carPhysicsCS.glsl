#version 430

layout(local_size_x = 1) in;

/*
    Car data:
    0: x
    1: y
    2: angle
    3: speed
    4: acceleration

    Car inputs:
    0: forward
    1: backward
    2: left
    3: right
    4: brake
*/

layout(binding = 0) buffer inputBuffer1 { float carData[]; };
layout(binding = 1) buffer inputBuffer2 { float carInputs[]; };
layout(binding = 1) buffer outputBuffer {float carOutputs[]; };

uniform int numCarFloats;
uniform int numInputs;

uniform float engineForce;
uniform float brakeForce;
uniform float maxTurnRate;
uniform float maxSpeed;
uniform float carMass;
uniform float deltaTime;


void main()
{
    uint index = gl_GlobalInvocationID.x;
    uint in1Index = index * numCarFloats;
    uint in2Index = index * numInputs;

    float x = carData[in1Index];
    float y = carData[in1Index + 1];
    float angle = carData[in1Index + 2];
    float speed = carData[in1Index + 3];
    float acceleration = carData[in1Index + 4];

    float appliedForce = 0.0;
    float appliedTurning = 0.0;

    appliedForce += carInputs[in2Index] * engineForce;
    appliedForce -= carInputs[in2Index + 1] * engineForce;
    appliedForce += carInputs[in2Index + 4] * brakeForce * (speed > 0.0 ? -1.0 : 1.0);
    appliedTurning += carInputs[in2Index + 2] * maxTurnRate;
    appliedTurning -= carInputs[in2Index + 3] * maxTurnRate;

    float vvmaxS = (speed / maxSpeed) * (speed / maxSPeed);

    float airResistance = vvmaxS * engineForce;
    float totalForce = appliedForce - airResistance;

    float totalTurning = appliedTurning * (1 - vvmaxS);

    
    acceleration = totalForce / carMass;
    angle += totalTurning * deltaTime;
    speed += car->acceleration * deltaTime;
    x += speed * cos(angle) * deltaTime;
    y += speed * sin(angle) * deltaTime;

    carData[in1Index] = x;
    carData[in1Index + 1] = y;
    carData[in1Index + 2] = angle;
    carData[in1Index + 3] = speed;
    carData[in1Index + 4] = acceleration;
}