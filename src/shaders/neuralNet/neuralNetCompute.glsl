#version 430

VARIABLE_WORKGROUP_SIZE

layout(binding = 0) buffer buffer1 { float inputs[]; };
layout(binding = 1) buffer buffer2 { float weights[]; };
layout(binding = 2) buffer buffer3 { float outputs[]; };

uniform int numOutputs;
uniform int numInputs;
uniform int offset;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main()
{
    uint network = gl_WorkGroupID.x + offset;
    uint outNode = gl_LocalInvocationID.x;

    uint inputOffset = network * numInputs;
    uint weightsOffset = network * (numInputs + 1) * numOutputs + outNode * (numInputs + 1);
    uint outputOffset = network * numOutputs + outNode;

    // Calculate the weighted sum of the inputs to the hidden layer
    float weightedSum = 0.0;
    for (int i = 0; i < numInputs; i++) {
        weightedSum += inputs[inputOffset + i] * weights[weightsOffset + i];
    }

    // Add the bias
    weightedSum += weights[weightsOffset + numInputs];

    // Apply the activation function
    outputs[outputOffset] = sigmoid(weightedSum);
}