#version 430

layout (local_size_x = 1) in;

layout (binding = 0) buffer inputBuffer1 { float seeds[]; };
layout (binding = 1) buffer outputBuffer1 {float layer1Weights[]; };
layout (binding = 2) buffer outputBuffer2 {float layer2Weights[]; };
layout (binding = 3) buffer outputBuffer3 {float layer3Weights[]; };
layout (binding = 4) buffer outputBuffer4 {float outputLayerWeights[]; };

uniform int topIndices[7];
uniform int numLeaders;
uniform int wheelChoices[3];
uniform int numWheelChoices;

uniform int numInputs;
uniform int numHiddenLayer1Nodes;
uniform int numHiddenLayer2Nodes;
uniform int numHiddenLayer3Nodes;
uniform int numOutputs;

uniform float learningRate;

// Hash function to generate pseudorandom numbers
float hash(float seed) {
    return fract(sin(seed) * 43758.5453123);
}

// Function to generate a random float in the range [-1, 1]
vec2 randomFloatInRange(float seed) {
    float randomValue = hash(seed) * 2.0 - 1.0; // Normalize to range [-1, 1]
    float newSeed = seed + 1.0;                 // Increment the seed for next use
    return vec2(randomValue, newSeed);
}

void main()
{
    uint index = gl_GlobalInvocationID.x;
    float seed = seeds[index];

    uint leaderIndexIndex = index % (numLeaders + numWheelChoices);

    int leaderIndex;
    if (leaderIndexIndex < numLeaders) {
        leaderIndex = topIndices[leaderIndexIndex];
    } else {
        leaderIndex = wheelChoices[leaderIndexIndex - numLeaders];
    }

    bool cont = true;
    for (int i = 0; i < numLeaders; i++) {
        if (topIndices[i] == int(index)) {
            cont = false;
        }
    }
    for (int i = 0; i < numWheelChoices; i++) {
        if (wheelChoices[i] == int(index)) {
            cont = false;
        }
    }

    if (!cont) {
        return;
    } else {

        // Generate random weights for the first hidden layer
        uint layerOffsetIndex = index * (numInputs + 1) * numHiddenLayer1Nodes;
        uint leaderOffsetIndex = leaderIndex * (numInputs + 1) * numHiddenLayer1Nodes;
        for (int i = 0; i < (numInputs + 1) * numHiddenLayer1Nodes; i++) {
            vec2 randomValue = randomFloatInRange(seed);
            layer1Weights[layerOffsetIndex + i] = layer1Weights[leaderOffsetIndex + i] + randomValue.x * learningRate;
            seed = randomValue.y;
        }

        // Generate random weights for the second hidden layer
        layerOffsetIndex = index * (numHiddenLayer1Nodes + 1) * numHiddenLayer2Nodes;
        leaderOffsetIndex = leaderIndex * (numHiddenLayer1Nodes + 1) * numHiddenLayer2Nodes;
        for (int i = 0; i < (numHiddenLayer1Nodes + 1) * numHiddenLayer2Nodes; i++) {
            vec2 randomValue = randomFloatInRange(seed);
            layer2Weights[layerOffsetIndex + i] = layer2Weights[leaderOffsetIndex + i] + randomValue.x * learningRate;
            seed = randomValue.y;
        }

        // Generate random weights for the third hidden layer
        layerOffsetIndex = index * (numHiddenLayer2Nodes + 1) * numHiddenLayer3Nodes;
        leaderOffsetIndex = leaderIndex * (numHiddenLayer2Nodes + 1) * numHiddenLayer3Nodes;
        for (int i = 0; i < (numHiddenLayer2Nodes + 1) * numHiddenLayer3Nodes; i++) {
            vec2 randomValue = randomFloatInRange(seed);
            layer3Weights[layerOffsetIndex + i] = layer3Weights[leaderOffsetIndex + i] + randomValue.x * learningRate;
            seed = randomValue.y;
        }

        // Generate random weights for the output layer
        layerOffsetIndex = index * (numHiddenLayer3Nodes + 1) * numOutputs;
        leaderOffsetIndex = leaderIndex * (numHiddenLayer3Nodes + 1) * numOutputs;
        for (int i = 0; i < (numHiddenLayer3Nodes + 1) * numOutputs; i++) {
            vec2 randomValue = randomFloatInRange(seed);
            outputLayerWeights[layerOffsetIndex + i] = outputLayerWeights[leaderOffsetIndex + i] + randomValue.x * learningRate;
            seed = randomValue.y;
        }
        seeds[index] = seed;
    }
}
