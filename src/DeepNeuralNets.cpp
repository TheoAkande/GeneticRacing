#include "DeepNeuralNets.h"
#include "Utils.h"

#ifndef DONT_USE_NNS

// General
int DeepNeuralNets::epoch = 0;
int DeepNeuralNets::lastCalculatedLeaders = -1;

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
GLuint DeepNeuralNets::inputs;

// Neural network weights
float DeepNeuralNets::layer1Weights[(NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::layer2Weights[(NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::layer3Weights[(NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::outputWeights[(NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * NUM_NEURAL_NETS];

// Neural network outputs
float DeepNeuralNets::layer1Outputs[NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::layer2Outputs[NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::layer3Outputs[NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::outputOutputs[NUM_OUTPUTS * NUM_NEURAL_NETS];

// Neural network fitness
GLuint DeepNeuralNets::fitnessSSBO;
float DeepNeuralNets::fitness[NUM_NEURAL_NETS];
int DeepNeuralNets::topIndices[NUM_GENERATION_LEADERS];
int DeepNeuralNets::wheelChoices[NUM_WHEEL_CHOICES];

// Generation leaders
float DeepNeuralNets::genLeadersLayer1Weights[(NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersLayer2Weights[(NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersLayer3Weights[(NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersOutputWeights[(NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersFitness[NUM_GENERATION_LEADERS];

// Private methods

// Create a population of randomly weighted neural nets
void DeepNeuralNets::createRandomPopulation(void) {
    // Setup Compute Shader
    glUseProgram(DeepNeuralNets::randomPopulationComputeShader);

    // Set the seeds
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * NUM_NEURAL_NETS, DeepNeuralNets::seeds, GL_DYNAMIC_DRAW);

    // Set the output buffers
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);

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

// Calculate the top fitness scores and indices of a generation
void DeepNeuralNets::calculateGenerationLeaderIndices(void) {
    // Gather fitness data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::fitnessSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * NUM_NEURAL_NETS, DeepNeuralNets::fitness);

    // Calculate the top fitness scores and indices
    float topFitness[NUM_GENERATION_LEADERS];
    DeepNeuralNets::topIndices[NUM_GENERATION_LEADERS];
    for (int i = 0; i < NUM_GENERATION_LEADERS; i++) {
        topFitness[i] = -1000000.0f;
        topIndices[i] = 0;
    }

    // Not so efficient - improve later
    for (int i = 0; i < NUM_NEURAL_NETS; i++) {
        float curFitness = DeepNeuralNets::fitness[i];
        float curIndex = i;
        for (int j = 0; j < NUM_GENERATION_LEADERS; j++) {
            if (curFitness > topFitness[j]) {
                float temp = curFitness;
                curFitness = topFitness[j];
                topFitness[j] = temp;
                int t = topIndices[j];
                topIndices[j] = curIndex;
                curIndex = t;
            }
        }
    }

    cout << "Epoch " << epoch + 1 << " best score: " << topFitness[0] << endl;

    DeepNeuralNets::lastCalculatedLeaders = DeepNeuralNets::epoch;
}

void DeepNeuralNets::spinWheel(void) {
    // Calculate Wheel indices
    int offset = NUM_NEURAL_NETS / NUM_WHEEL_CHOICES;
    for (int i = 0; i < NUM_WHEEL_CHOICES; i++) {
        DeepNeuralNets::wheelChoices[i] = i * offset + (rand() % offset);
    }
}

// Export a given model to a file
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
    for (int i = 0; i < (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES; i++) {
        file << layer1Weights[i] << endl;
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES; i++) {
        file << layer2Weights[i] << endl;
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES; i++) {
        file << layer3Weights[i] << endl;
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS; i++) {
        file << outputWeights[i] << endl;
    }

    file.close();
}

// Export the best model of the generation
void DeepNeuralNets::exportPopulationModel(string filename, int index) {
    // Gather the weights
    float *layer1WeightAddress = &DeepNeuralNets::layer1Weights[index * (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[1]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, index * (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * sizeof(float), (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * sizeof(float), layer1WeightAddress);
    float *layer2WeightAddress = &DeepNeuralNets::layer2Weights[index * (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[2]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, index * (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * sizeof(float), (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * sizeof(float), layer2WeightAddress);
    float *layer3WeightAddress = &DeepNeuralNets::layer3Weights[index * (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[3]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, index * (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * sizeof(float), (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * sizeof(float), layer3WeightAddress);
    float *outputWeightAddress = &DeepNeuralNets::outputWeights[index * (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[4]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, index * (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * sizeof(float), (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * sizeof(float), outputWeightAddress);

    // Export the model
    DeepNeuralNets::exportModel(filename, layer1WeightAddress, layer2WeightAddress, layer3WeightAddress, outputWeightAddress);
}

// Export a model from the generation leaders
void DeepNeuralNets::exportTopModel(string filename, int index) {
    DeepNeuralNets::exportModel(
        filename, 
        &DeepNeuralNets::genLeadersLayer1Weights[index * (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES], 
        &DeepNeuralNets::genLeadersLayer2Weights[index * (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES], 
        &DeepNeuralNets::genLeadersLayer3Weights[index * (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES], 
        &DeepNeuralNets::genLeadersOutputWeights[index * (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS]
    );
}

// Public methods

DeepNeuralNets::DeepNeuralNets() {
    // Empty constructor
}

// Setup the training of the neural nets
void DeepNeuralNets::setupTraining(GLuint carData, GLuint computerVisionData, GLuint inputs, GLuint fitness) {
    DeepNeuralNets::initNeuralNets(carData, computerVisionData, inputs, fitness);
    DeepNeuralNets::createRandomPopulation();
}

// Initialize the neural nets with the SSBOs (does not create random population) - ie for testing/usage
void DeepNeuralNets::initNeuralNets(GLuint carData, GLuint computerVisionData, GLuint inputs, GLuint fitness) {
    // Set the pointers to the input buffers
    DeepNeuralNets::carData = carData;
    DeepNeuralNets::computerVisionData = computerVisionData;
    DeepNeuralNets::inputs = inputs;
    DeepNeuralNets::fitnessSSBO = fitness;

    // Create the compute shaders
    DeepNeuralNets::Layer1ComputeShader = Utils::createShaderProgram("shaders/neuralNet/layer1CS.glsl", NUM_HIDDEN_LAYER_1_NODES);
    DeepNeuralNets::Layer2ComputeShader = Utils::createShaderProgram("shaders/neuralNet/neuralNetCompute.glsl", NUM_HIDDEN_LAYER_2_NODES);
    DeepNeuralNets::Layer3ComputeShader = Utils::createShaderProgram("shaders/neuralNet/neuralNetCompute.glsl", NUM_HIDDEN_LAYER_3_NODES);
    DeepNeuralNets::OutputComputeShader = Utils::createShaderProgram("shaders/neuralNet/neuralNetCompute.glsl", NUM_OUTPUTS);

    DeepNeuralNets::evolutionComputeShader = Utils::createShaderProgram("shaders/neuralNet/evolutionCompute.glsl");
    
    DeepNeuralNets::randomPopulationComputeShader = Utils::createShaderProgram("shaders/neuralNet/randomPopulationCompute.glsl");
    

    // Create the neural net compute buffer objects
    glGenBuffers(NUM_NN_CBS, nnCBOs);

    // Set the seeds (note: might change to ints later)
    for (int i = 0; i < NUM_NEURAL_NETS; i++) {
        DeepNeuralNets::seeds[i] = (float)rand();
    }

    // Setup nnCBO[5-7] for the outputs
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[5]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[6]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[7]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS, NULL, GL_DYNAMIC_DRAW);
}

// Feed inputs through the NNs
// Note: we don't use iteration because "VARIABLE_WORKGROUP_SIZE" means we have different shader programs
//       for each layer to improve performance
void DeepNeuralNets::invokeNeuralNets(glm::vec4 startLine) {
    // Calculate layer 1
    glUseProgram(DeepNeuralNets::Layer1ComputeShader);

    GLuint uLoc = glGetUniformLocation(DeepNeuralNets::Layer1ComputeShader, "numInputs");
    glUniform1i(uLoc, NUM_INPUTS);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer1ComputeShader, "numOutputs");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_1_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer1ComputeShader, "numHiddenLayerNodes");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_1_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer1ComputeShader, "startLine");
    glUniform4f(uLoc, startLine.x, startLine.y, startLine.z, startLine.w);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer1ComputeShader, "numCarFloats");
    glUniform1i(uLoc, numCarFloats);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer1ComputeShader, "numVisionFloats");
    glUniform1i(uLoc, numComputerVisionAngles);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer1ComputeShader, "numDrivers");
    glUniform1i(uLoc, numDrivers);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, DeepNeuralNets::carData);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, DeepNeuralNets::computerVisionData);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, DeepNeuralNets::nnCBOs[1]); // layer1Weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, DeepNeuralNets::nnCBOs[5]); // layer1Outputs

    glDispatchCompute(NUM_NEURAL_NETS, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    // Calculate layer 2
    glUseProgram(DeepNeuralNets::Layer2ComputeShader);

    uLoc = glGetUniformLocation(DeepNeuralNets::Layer2ComputeShader, "numInputs");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_1_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer2ComputeShader, "numOutputs");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_2_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer2ComputeShader, "offset");
    glUniform1i(uLoc, 0);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, DeepNeuralNets::nnCBOs[5]); // layer1Outputs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, DeepNeuralNets::nnCBOs[2]); // layer2Weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, DeepNeuralNets::nnCBOs[6]); // layer2Outputs

    glDispatchCompute(NUM_NEURAL_NETS, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    // Calculate layer 3
    glUseProgram(DeepNeuralNets::Layer3ComputeShader);

    uLoc = glGetUniformLocation(DeepNeuralNets::Layer3ComputeShader, "numInputs");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_2_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer3ComputeShader, "numOutputs");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_3_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer2ComputeShader, "offset");
    glUniform1i(uLoc, 0);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, DeepNeuralNets::nnCBOs[6]); // layer2Outputs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, DeepNeuralNets::nnCBOs[3]); // layer3Weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, DeepNeuralNets::nnCBOs[7]); // layer3Outputs

    glDispatchCompute(NUM_NEURAL_NETS, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    // Calculate output
    glUseProgram(DeepNeuralNets::OutputComputeShader);

    uLoc = glGetUniformLocation(DeepNeuralNets::OutputComputeShader, "numInputs");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_3_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::OutputComputeShader, "numOutputs");
    glUniform1i(uLoc, NUM_OUTPUTS);
    uLoc = glGetUniformLocation(DeepNeuralNets::Layer2ComputeShader, "offset");
    glUniform1i(uLoc, numDrivers);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, DeepNeuralNets::nnCBOs[7]); // layer3Outputs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, DeepNeuralNets::nnCBOs[4]); // outputWeights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, DeepNeuralNets::inputs);    // car inputs

    glDispatchCompute(NUM_NEURAL_NETS, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

// Gather the weights of the top models into the generation leaders
void DeepNeuralNets::gatherGenerationLeaders(void) {
    // Calculate the generation leaders
    if (lastCalculatedLeaders != epoch) {
        DeepNeuralNets::calculateGenerationLeaderIndices();
    }

    // Gather the weights of the top models into the generation leaders
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[1]);
    for (int i = 0; i < NUM_GENERATION_LEADERS; i++) {
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, topIndices[i] * (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * sizeof(float), (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * sizeof(float), &DeepNeuralNets::genLeadersLayer1Weights[i * (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES]);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[2]);
    for (int i = 0; i < NUM_GENERATION_LEADERS; i++) {
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, topIndices[i] * (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * sizeof(float), (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * sizeof(float), &DeepNeuralNets::genLeadersLayer2Weights[i * (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES]);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[3]);
    for (int i = 0; i < NUM_GENERATION_LEADERS; i++) {
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, topIndices[i] * (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * sizeof(float), (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * sizeof(float), &DeepNeuralNets::genLeadersLayer3Weights[i * (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES]);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[4]);
    for (int i = 0; i < NUM_GENERATION_LEADERS; i++) {
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, topIndices[i] * (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * sizeof(float), (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * sizeof(float), &DeepNeuralNets::genLeadersOutputWeights[i * (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS]);
    }
}

// Apply the genetic algorithm to evolve the neural nets
void DeepNeuralNets::evolveNeuralNets(void) {
    // Calculate gen leader indices
    if (DeepNeuralNets::lastCalculatedLeaders != DeepNeuralNets::epoch) {
        DeepNeuralNets::calculateGenerationLeaderIndices();
    }

    // Calculate the wheel choices
    DeepNeuralNets::spinWheel();

    // Evolve the generation leaders
    glUseProgram(DeepNeuralNets::evolutionComputeShader);

    // Set the generation leaders
    GLuint uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "topIndices");
    glUniform1iv(uLoc, NUM_GENERATION_LEADERS, DeepNeuralNets::topIndices);
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "numLeaders");
    glUniform1i(uLoc, NUM_GENERATION_LEADERS);
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "numWheelChoices");
    glUniform1i(uLoc, NUM_WHEEL_CHOICES);
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "wheelChoices");
    glUniform1iv(uLoc, NUM_WHEEL_CHOICES, DeepNeuralNets::wheelChoices);

    // Set the layer sizes
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "numInputs");
    glUniform1i(uLoc, NUM_INPUTS);
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "numHiddenLayer1Nodes");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_1_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "numHiddenLayer2Nodes");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_2_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "numHiddenLayer3Nodes");
    glUniform1i(uLoc, NUM_HIDDEN_LAYER_3_NODES);
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "numOutputs");
    glUniform1i(uLoc, NUM_OUTPUTS);

    // Set the learning rate
    uLoc = glGetUniformLocation(DeepNeuralNets::evolutionComputeShader, "learningRate");
    glUniform1f(uLoc, LEARNING_RATE);

    // Set the buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, DeepNeuralNets::nnCBOs[0]); // seeds
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, DeepNeuralNets::nnCBOs[1]); // layer1Weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, DeepNeuralNets::nnCBOs[2]); // layer2Weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, DeepNeuralNets::nnCBOs[3]); // layer3Weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, DeepNeuralNets::nnCBOs[4]); // outputWeights

    // Dispatch the compute shader
    glDispatchCompute(NUM_NEURAL_NETS, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    DeepNeuralNets::epoch++;
}

// Export the best model of the generation
void DeepNeuralNets::exportBestModel(void) {
    DeepNeuralNets::gatherGenerationLeaders();
    string leaderFilename = 
        "../../../src/assets/models/epoch" + 
        to_string(DeepNeuralNets::epoch) + "_" + 
        "best.txt";
    DeepNeuralNets::exportTopModel(leaderFilename, 0);
}

// Export the generation leaders
void DeepNeuralNets::exportGenerationLeaders(void) {
    DeepNeuralNets::gatherGenerationLeaders();
    // Export the generation leaders
    for (int i = 0; i < NUM_GENERATION_LEADERS; i++) {
        string leaderFilename = 
            "../../../src/assets/models/epoch" + 
            to_string(DeepNeuralNets::epoch) + "_" + 
            "top_" +
            to_string(i + 1) + ".txt";
        DeepNeuralNets::exportTopModel(leaderFilename, i);
    }
}

// Import a model from a file
void DeepNeuralNets::importModel(string filename, int index) {
    // Import the model
    ifstream file;
    file.open(filename);

    if (!file.is_open()) {
        cout << "Error opening file" << endl;
        return;
    }

    // Read the weights
    for (int i = 0; i < (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES; i++) {
        file >> DeepNeuralNets::layer1Weights[index * (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES + i];
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES; i++) {
        file >> DeepNeuralNets::layer2Weights[index * (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES + i];
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES; i++) {
        file >> DeepNeuralNets::layer3Weights[index * (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES + i];
    }
    for (int i = 0; i < (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS; i++) {
        file >> DeepNeuralNets::outputWeights[index * (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS + i];
    }

    file.close();

    // Update the weights
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS, DeepNeuralNets::layer1Weights, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS, DeepNeuralNets::layer2Weights, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS, DeepNeuralNets::layer3Weights, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DeepNeuralNets::nnCBOs[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * NUM_NEURAL_NETS, DeepNeuralNets::outputWeights, GL_DYNAMIC_DRAW);
}

#endif