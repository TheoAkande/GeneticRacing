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
float *DeepNeuralNets::carData;
float *DeepNeuralNets::computerVisionData;
float *DeepNeuralNets::startLine;

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

// Public methods

DeepNeuralNets::DeepNeuralNets() {
    // Empty constructor
}

void DeepNeuralNets::initNeuralNets(float *carData, float *computerVisionData, float *startLine) {
    // Set the pointers to the input buffers
    DeepNeuralNets::carData = carData;
    DeepNeuralNets::computerVisionData = computerVisionData;
    DeepNeuralNets::startLine = startLine;

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

    auto start = std::chrono::high_resolution_clock::now();
    DeepNeuralNets::createRandomPopulation();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
    while (true) {
        // Placeholder
        
    }
}