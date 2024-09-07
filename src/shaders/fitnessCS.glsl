#version 430

layout(local_size_x = 1) in;

layout (binding = 0) buffer inputBuffer1 { float carData[]; };
layout (binding = 1) buffer inputBuffer2 { float carEval[]; };
layout (binding = 2) buffer inputBuffer3 {float gates[]; };
layout (binding = 3) buffer inputBuffer4 {float vision[]; };
layout (binding = 4) buffer outputBuffer {float fitness[]; };

uniform int numCarFloats;
uniform int numEvalFloats;
uniform int numGates;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    
    // Number of gates passed
    float gatesPassed = float(int(carEval[index * numEvalFloats + 1])) - 1;

    int gateIndex = int(carEval[index * numEvalFloats]);
    int lastGate = gateIndex == 0 ? numGates - 1 : gateIndex - 1;

    // Distance from last gate to next
    float distanceBetweenGates = length(
        vec2(
            gates[gateIndex * 3] - gates[lastGate * 3],
            gates[gateIndex * 3 + 1] - gates[lastGate * 3 + 1]
        )
    );

    // Distance from car to next gate
    float distanceToNextGate = length(
        vec2(
            gates[gateIndex * 3] - carData[index * numCarFloats],
            gates[gateIndex * 3 + 1] - carData[index * numCarFloats + 1]
        )
    );

    // Interpolate distance to next gate
    // Note: will be negative if further away that last gate. That is intended
    float interpolatedDistance = distanceToNextGate / distanceBetweenGates;
    float distanceFitness = 1.0 - interpolatedDistance;
    
    // Fitness is interpolated number of laps completed
    fitness[index] = (gatesPassed + distanceFitness) / float(numGates);
}