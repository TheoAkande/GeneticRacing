#include "DeepNeuralNets.h"
#include "Utils.h"

// Compute shader variables
GLuint 
    DeepNeuralNets::Layer1ComputeShader, 
    DeepNeuralNets::Layer2ComputeShader, 
    DeepNeuralNets::Layer3ComputeShader, 
    DeepNeuralNets::OutputComputeShader;
GLuint DeepNeuralNets::evolutionComputeShader;
GLuint DeepNeuralNets::randomPopulationComputeShader;
GLuint DeepNeuralNets::nnCBOs[NUM_NN_CBS];

// Network random seeds
float DeepNeuralNets::seeds[NUM_NEURAL_NETS];

// Network inputs
GLuint DeepNeuralNets::carData;
GLuint DeepNeuralNets::computerVisionData;
GLuint DeepNeuralNets::startLine;

// Neural network weights
float DeepNeuralNets::layer1Weights[(NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_NEURAL_NETS];
float DeepNeuralNets::layer2Weights[(NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_NEURAL_NETS];
float DeepNeuralNets::layer3Weights[(NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_NEURAL_NETS];
float DeepNeuralNets::outputWeights[(NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * NUM_NEURAL_NETS];

// Neural network outputs
float DeepNeuralNets::layer1Outputs[NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::layer2Outputs[NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::layer3Outputs[NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::outputOutputs[NUM_OUTPUTS * NUM_NEURAL_NETS];

// Neural network fitness
GLuint DeepNeuralNets::fitnessSSBO;
float DeepNeuralNets::fitness[NUM_NEURAL_NETS];

// Generation leaders
float DeepNeuralNets::genLeadersLayer1Weights[(NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersLayer2Weights[(NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersLayer3Weights[(NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersOutputWeights[(NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersFitness[NUM_GENERATION_LEADERS];

// Private methods

void DeepNeuralNets::createRandomPopulation(void) {
    // Setup Compute Shader
    glUseProgram(DeepNeuralNets::randomPopulationComputeShader);

    // Set the seeds
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * NUM_NEURAL_NETS, DeepNeuralNets::seeds, GL_DYNAMIC_DRAW);

    // Set the output buffers
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, DeepNeuralNets::nnCBOs[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, DeepNeuralNets::nnCBOs[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, DeepNeuralNets::nnCBOs[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, DeepNeuralNets::nnCBOs[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, DeepNeuralNets::nnCBOs[4]);

    // Set the uniforms
    GLuint uLoc = glGetUniformLocation(DeepNeuralNets::randomPopulationComputeShader, "numInputs");
    glUniform1i(uLoc, NUM_INPUTS);
    uLoc = glGetUniformLocation(DeepNeuralNets::randomPopulationComputeShader, "numHiddenLayer1Nodes");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_1_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::randomPopulationComputeShader, "numHiddenLayer2Nodes");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_2_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::randomPopulationComputeShader, "numHiddenLayer3Nodes");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_3_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::randomPopulationComputeShader, "numOutputs");
    glUniform1i(uLoc, NUM_OUTPUTS);
    uLoc = glGetUniformLocation(DeepNeuralNets::randomPopulationComputeShader, "numNeuralNets");
    glUniform1i(uLoc, NUM_NEURAL_NETS);

    // Dispatch the compute shader
    glDispatchCompute(NUM_NEURAL_NETS, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void DeepNeuralNets::exportModel(string filename, float *layer1Weights, float *layer2Weights, float *layer3Weights, float *outputWeights) {
    cout << "Exporting model to " << filename << endl;
    
    // Export the model
    ofstream file;
    file.open(filename);

    if (!file.is_open()) {
        cout << "Error opening file" << endl;
        return;
    }

    // Write the weights
    for (int i = 0; i < (NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1); i++) {
        file << layer1Weights[i] << endl;
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1); i++) {
        file << layer2Weights[i] << endl;
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1); i++) {
        file << layer3Weights[i] << endl;
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1); i++) {
        file << outputWeights[i] << endl;
    }

    file.close();
}

void DeepNeuralNets::exportPopulationModel(string filename, int index) {
    // Gather the weights
    float *layer1WeightAddress = &DeepNeuralNets::layer1Weights[index * (NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1)];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[1]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, index * (NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * sizeof(float), (NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * sizeof(float), layer1WeightAddress);
    float *layer2WeightAddress = &DeepNeuralNets::layer2Weights[index * (NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1)];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[2]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, index * (NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * sizeof(float), (NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * sizeof(float), layer2WeightAddress);
    float *layer3WeightAddress = &DeepNeuralNets::layer3Weights[index * (NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1)];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[3]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, index * (NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * sizeof(float), (NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * sizeof(float), layer3WeightAddress);
    float *outputWeightAddress = &DeepNeuralNets::outputWeights[index * (NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1)];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[4]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, index * (NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * sizeof(float), (NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * sizeof(float), outputWeightAddress);

    // Export the model
    DeepNeuralNets::exportModel(filename, layer1WeightAddress, layer2WeightAddress, layer3WeightAddress, outputWeightAddress);
}

// Public methods

DeepNeuralNets::DeepNeuralNets() {
    // Empty constructor
}

void DeepNeuralNets::initNeuralNets(GLuint carData, GLuint computerVisionData, GLuint startLine, GLuint fitness) {
    // Set the pointers to the input buffers
    DeepNeuralNets::carData = carData;
    DeepNeuralNets::computerVisionData = computerVisionData;
    DeepNeuralNets::startLine = startLine;
    DeepNeuralNets::fitnessSSBO = fitness;

    // Create the compute shaders
    /*
    DeepNeuralNets::Layer1ComputeShader = Utils::createShaderProgram("shaders/neuralNet/neuralNetCompute.glsl", NUM_HIDDEN_LAYER_1_NODES);
    DeepNeuralNets::Layer2ComputeShader = Utils::createShaderProgram("shaders/neuralNet/neuralNetCompute.glsl", NUM_HIDDEN_LAYER_2_NODES);
    DeepNeuralNets::Layer3ComputeShader = Utils::createShaderProgram("shaders/neuralNet/neuralNetCompute.glsl", NUM_HIDDEN_LAYER_3_NODES);
    DeepNeuralNets::OutputComputeShader = Utils::createShaderProgram("shaders/neuralNet/neuralNetCompute.glsl", NUM_OUTPUTS);
    DeepNeuralNets::evolutionComputeShader = Utils::createShaderProgram("shaders/neuralNet/evolutionCompute.glsl");
    */
    DeepNeuralNets::randomPopulationComputeShader = Utils::createShaderProgram("shaders/neuralNet/randomPopulationCompute.glsl");

    // Create the neural net compute buffer objects
    glGenBuffers(NUM_NN_CBS, nnCBOs);

    // Set the seeds (note: might change to ints later)
    for (int i = 0; i < NUM_NEURAL_NETS; i++) {
        DeepNeuralNets::seeds[i] = (float)rand();
    }

    DeepNeuralNets::createRandomPopulation();
    DeepNeuralNets::exportPopulationModel("../../../src/assets/models/populationModel0.txt", 0);
    DeepNeuralNets::exportPopulationModel("../../../src/assets/models/populationModel1.txt", 1);
}