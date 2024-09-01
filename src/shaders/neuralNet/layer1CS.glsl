#version 430

VARIABLE_WORKGROUP_SIZE

layout(binding = 0) buffer buffer1 { float carData[]; };
layout(binding = 1) buffer buffer2 { float visionData[]; };
layout(binding = 2) buffer buffer3 { float layer1Weights[]; };
layout(binding = 3) buffer buffer4 { float layer1Outputs[]; };

uniform vec4 startLine;
uniform int numCarFloats;
uniform int numVisionFloats;
uniform int numHiddenLayerNodes;
uniform int numOutputs;
uniform int numInputs;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main()
{
    uint network = gl_WorkGroupID.x;
    uint outNode = gl_LocalInvocationID.x;

    uint carDataOffset = network * numCarFloats;
    uint visionDataOffset = network * numVisionFloats;
    uint layerWeightsOffset = network * (numInputs + 1) * numHiddenLayerNodes + outNode * (numInputs + 1);
    uint layerOutputsOffset = network * numHiddenLayerNodes + outNode;

    // Calculate the weighted sum of the inputs to the hidden layer
    float weightedSum = 0.0;
    for (int i = 0; i < numCarFloats; i++) {
        weightedSum += carData[carDataOffset + i] * layer1Weights[layerWeightsOffset + i];
    }
    for (int i = 0; i < numVisionFloats; i++) {
        weightedSum += visionData[visionDataOffset + i] * layer1Weights[layerWeightsOffset + numCarFloats + i];
    }
    weightedSum += startLine.x * layer1Weights[layerWeightsOffset + numCarFloats + numVisionFloats];
    weightedSum += startLine.y * layer1Weights[layerWeightsOffset + numCarFloats + numVisionFloats + 1];
    weightedSum += startLine.z * layer1Weights[layerWeightsOffset + numCarFloats + numVisionFloats + 2];
    weightedSum += startLine.w * layer1Weights[layerWeightsOffset + numCarFloats + numVisionFloats + 3];

    // Add the bias
    weightedSum += layer1Weights[layerWeightsOffset + numInputs];

    // Apply the activation function
    layer1Outputs[layerOutputsOffset] = sigmoid(weightedSum);
}