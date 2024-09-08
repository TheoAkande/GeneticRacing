#version 430

layout (local_size_x = 1) in;

layout(binding = 0) buffer buffer1 { float weights[]; };
layout(binding = 1) buffer buffer2 { float inputs[]; };
layout(binding = 2) buffer buffer3 { float outputs[]; };

uniform int numInputs;
uniform int numOutputs;

float ReLU(float x) {
    return max(0.0, x);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main()
{
    uint index = gl_GlobalInvocationID.x;
    uint weightsBase = index * (numInputs + 1);

    // Calculate the weighted sum of the inputs to the hidden layer
    float weightedSum = 0.0;
    for (int i = 0; i < numInputs; i++) {
        weightedSum += inputs[i] * weights[weightsBase +  i];
    }

    // Add the bias
    weightedSum += weights[weightsBase + numInputs];

    // Apply the activation function
    outputs[index] = ReLU(weightedSum);
}