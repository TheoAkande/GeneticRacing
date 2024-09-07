extern "C" {
    _declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <SOIL2/soil2.h>
#include <cstdlib>
#include <time.h>

#include "Utils.h"
#include "TrackMaker.h"
#include "Defs.h"
#include "DeepNeuralNets.h"

using namespace std;

GLuint vao[numVAOs];
GLuint vbo[numVBOs];
GLuint cbo[numCBs];
GLuint vcbo[numVCBs];
GLuint cwLoc, chLoc, ncfLoc, colLoc;
GLuint trackRenderingProgram, carRenderingProgram, wheelComputeShader, 
    physicsComputeShader, driverRenderingProgram, tsRenderingProgram,
    computerVisionComputeShader, computerVisionRenderingProgram,
    fittnessComputeShader;

float inputs[numInputs * numCars];
float carPos[numCars * numCarFloats];
float carPoints[numCars * 5 * 2];

float fitness[numCars];
float carEvalData[numCars * numCarEvalFloats];
int frames = 0;

float computerVisionAngles[] = {
    0.0f, // straight ahead
    0.785398f, // 45 degrees
    -0.785398f, // -45 degrees
    1.5708f, // 90 degrees
    -1.5708f, // -90 degrees
    0.1f, // 6 degrees
    -0.1f, // -6 degrees
    0.174533f, // 10 degrees
    -0.174533f, // -10 degrees
    0.698132f, // 40 degrees
    -0.698132f, // -40 degrees
    0.872665, // 50 degrees
    -0.872665, // -50 degrees
    2.35619, // 135 degrees
    -2.35619 // -135 degrees
};
float computerVisionDistances[numCars * numComputerVisionAngles];
bool showTrack = true;
bool sHeld = false;

int carInputs[] = {
    GLFW_KEY_UP,
    GLFW_KEY_DOWN,
    GLFW_KEY_LEFT,
    GLFW_KEY_RIGHT,
    GLFW_KEY_SPACE,
    GLFW_KEY_W,
    GLFW_KEY_S,
    GLFW_KEY_A,
    GLFW_KEY_D,
    GLFW_KEY_LEFT_SHIFT,
};

float carColours[] = {
    0.0f, 1.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f
};

double deltaTime = 0.0l;
double lastTime = 0.0l;

glm::mat4 viewMat;

float appliedForce, totalForce, airResistance;
float appliedTurning, totalTurning;

const char *track = "assets/tracks/track6.tr";
vector<float> insideTrack;
vector<float> outsideTrack;
vector<float> midpoints;
vector<float> midpointsWithAngles;
vector<float> normals;
int numGates = 0;
float trackStartLine[4];
glm::vec2 trackStartNormal;
float carX, carY, carAngle, carSpeed, carAcceleration; // angle 0 = right, 90 = up
GLuint efLoc, bfLoc, mtrLoc, msLoc, cmLoc, dtLoc, niLoc, nt1Loc, nt2Loc;

bool cHeld = false;

int numTracks;
int curTrack = 0;

bool shouldCreateTrack = false;

struct Car {
    float x, y, angle;
    float speed;
    float acceleration;
};

Car cars[numCars];

void calculateFitness(void) {
    glUseProgram(fittnessComputeShader);

    ncfLoc = glGetUniformLocation(fittnessComputeShader, "numCarFloats");
    glUniform1i(ncfLoc, numCarFloats);
    ncfLoc = glGetUniformLocation(fittnessComputeShader, "numEvalFloats");
    glUniform1i(ncfLoc, numCarEvalFloats);
    ncfLoc = glGetUniformLocation(fittnessComputeShader, "numGates");
    glUniform1i(ncfLoc, numGates);
    ncfLoc = glGetUniformLocation(fittnessComputeShader, "numComputerVisionAngles");
    glUniform1i(ncfLoc, numComputerVisionAngles);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo[0]); // car data
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cbo[8]); // car eval data
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cbo[7]); // gates
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo[5]); // computer vision distances
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cbo[1]); // fitness

    glDispatchCompute(numCars, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void renderComputerVision(void) {
    // Get computer vision distances
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[5]); // computerVisionDistances
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars * numComputerVisionAngles, &computerVisionDistances[0]);

    // Get car data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[0]); // carPos
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars * numCarFloats, &carPos[0]);


    glUseProgram(computerVisionRenderingProgram);

    float visionPoints[numCars * numComputerVisionAngles * 2];

    for (int i = 0; i < numCars; i++) {
        for (int j = 0; j < numComputerVisionAngles; j++) {
            float x = carPos[i * numCarFloats];
            float y = carPos[i * numCarFloats + 1];

            float angle = carPos[i * numCarFloats + 2] + computerVisionAngles[j];
            float distance = computerVisionDistances[i * numComputerVisionAngles + j];
            x += distance * cos(angle);
            y += distance * sin(angle);

            visionPoints[i * numComputerVisionAngles * 2 + j * 2] = x;
            visionPoints[i * numComputerVisionAngles * 2 + j * 2 + 1] = y;
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo[5]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * numCars * numComputerVisionAngles * 2, visionPoints, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_POINTS, 0, numCars * numComputerVisionAngles);
}

void calculateComputerVision(void) {
    glUseProgram(computerVisionComputeShader);

    ncfLoc = glGetUniformLocation(computerVisionComputeShader, "numCarFloats");
    glUniform1i(ncfLoc, numCarFloats);
    ncfLoc = glGetUniformLocation(computerVisionComputeShader, "numComputerVisionAngles");
    glUniform1i(ncfLoc, numComputerVisionAngles);
    nt1Loc = glGetUniformLocation(computerVisionComputeShader, "numInsidePoints");
    glUniform1i(nt1Loc, insideTrack.size() / 2);
    nt2Loc = glGetUniformLocation(computerVisionComputeShader, "numOutsidePoints");
    glUniform1i(nt2Loc, outsideTrack.size() / 2);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo[0]); // carPos
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cbo[3]); // insideTrack
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cbo[4]); // outsideTrack
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo[6]); // computerVisionAngles
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cbo[5]); // computerVisionDistances

    glDispatchCompute(numCars, numComputerVisionAngles, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void calculateCarPhysics(void) {
    glUseProgram(physicsComputeShader);

    ncfLoc = glGetUniformLocation(physicsComputeShader, "numCarFloats");
    glUniform1i(ncfLoc, numCarFloats);
    niLoc = glGetUniformLocation(physicsComputeShader, "numInputs");
    glUniform1i(niLoc, numInputs);
    ncfLoc = glGetUniformLocation(physicsComputeShader, "numEvalFloats"); 
    glUniform1i(ncfLoc, numCarEvalFloats);
    nt1Loc = glGetUniformLocation(physicsComputeShader, "numInsideTrackPoints");
    glUniform1i(nt1Loc, insideTrack.size() / 2);
    nt2Loc = glGetUniformLocation(physicsComputeShader, "numOutsideTrackPoints");
    glUniform1i(nt2Loc, outsideTrack.size() / 2);
    dtLoc = glGetUniformLocation(physicsComputeShader, "numGates");
    glUniform1i(dtLoc, numGates);

    efLoc = glGetUniformLocation(physicsComputeShader, "engineForce");
    glUniform1f(efLoc, carForce);
    bfLoc = glGetUniformLocation(physicsComputeShader, "brakeForce");
    glUniform1f(bfLoc, breakingForce);
    mtrLoc = glGetUniformLocation(physicsComputeShader, "maxTurnRate");
    glUniform1f(mtrLoc, maxTurningRate);
    msLoc = glGetUniformLocation(physicsComputeShader, "maxSpeed");
    glUniform1f(msLoc, vMax);
    cmLoc = glGetUniformLocation(physicsComputeShader, "carMass");
    glUniform1f(cmLoc, carMass);
    dtLoc = glGetUniformLocation(physicsComputeShader, "deltaTime");
    glUniform1f(dtLoc, deterministic ? deterministicDt : deltaTime);
    dtLoc = glGetUniformLocation(physicsComputeShader, "insideStart");
    glUniform2f(dtLoc, insideTrack[0], insideTrack[1]);
    dtLoc = glGetUniformLocation(physicsComputeShader, "outsideStart");
    glUniform2f(dtLoc, outsideTrack[0], outsideTrack[1]);
    dtLoc = glGetUniformLocation(physicsComputeShader, "startNormal");
    glUniform2f(dtLoc, trackStartNormal.x, trackStartNormal.y);
    dtLoc = glGetUniformLocation(physicsComputeShader, "startPoint");
    glUniform2f(dtLoc, cars[0].x, cars[0].y);
    dtLoc = glGetUniformLocation(physicsComputeShader, "startAngle");
    glUniform1f(dtLoc, cars[0].angle);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo[0]); // carPos
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cbo[2]); // inputs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cbo[3]); // insideTrack 
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo[4]); // outsideTrack
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cbo[8]); // car eval data
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, cbo[7]); // gata info

    glDispatchCompute(numCars, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void calculateCarWheels(void) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vcbo[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * 5 * 2, NULL, GL_STATIC_READ);

    glUseProgram(wheelComputeShader);

    ncfLoc = glGetUniformLocation(wheelComputeShader, "numCarFloats");
    glUniform1i(ncfLoc, numCarFloats);
    cwLoc = glGetUniformLocation(wheelComputeShader, "carWidth");
    glUniform1f(cwLoc, carWidth);
    chLoc = glGetUniformLocation(wheelComputeShader, "carHeight");
    glUniform1f(chLoc, carHeight);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vcbo[0]);

    glDispatchCompute(numCars, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vcbo[0]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars * 5 * 2, &carPoints[0]);

    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * numCars * 5 * 2, &carPoints[0], GL_STATIC_DRAW);
}

void loadCars(bool training) {
    for (int i = 0; i < numCars; i++) {
        carPos[i * numCarFloats] = cars[i].x;
        carPos[i * numCarFloats + 1] = cars[i].y;
        carPos[i * numCarFloats + 2] = cars[i].angle;
        carPos[i * numCarFloats + 3] = cars[i].speed;
        carPos[i * numCarFloats + 4] = cars[i].acceleration;
        carPos[i * numCarFloats + 5] = midpointsWithAngles[0] - cars[i].x; // next gate x
        carPos[i * numCarFloats + 6] = midpointsWithAngles[1] - cars[i].y; // next gate y
        carPos[i * numCarFloats + 7] = midpointsWithAngles[2] - cars[i].angle; // next gate angle


        carEvalData[i * numCarEvalFloats] = 0.0f; // start by passing start line
        carEvalData[i * numCarEvalFloats + 1] = 0.0f; // no gates passed yet
        carEvalData[i * numCarEvalFloats + 2] = 0.0f; // no speed yet

        fitness[i] = 0.0f; // no fitness yet (on this track)
    }

    if (!training) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(carColours), &carColours[0], GL_STATIC_DRAW);
    }

    // Bind carPos and fitness to compute buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numCarFloats, &carPos[0], GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars, &fitness[0], GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[8]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numCarEvalFloats, &carEvalData[0], GL_DYNAMIC_DRAW);

    if (!training) calculateCarWheels();
}

void loadTrack(string track, bool training = false) {
    // Load track from file
    insideTrack.clear();
    outsideTrack.clear();
    midpoints.clear();
    normals.clear();
    numGates = 0;
    ifstream fileStream(track, ios::in);
    string line = "";    
    bool track1 = true;
    bool started = false;
    float x, y;
    while (!fileStream.eof()) {
        getline(fileStream, line);
        if (line.c_str()[0] == '-') {
            track1 = false;
            started = false;
        } 

        if (line.c_str()[0] == 'n') {
            float n1, n2;
            sscanf(line.c_str(), "n %f %f %f %f", &x, &y, &n1, &n2);
            midpoints.push_back(x);
            midpoints.push_back(y);
            normals.push_back(n1);
            normals.push_back(n2);
            midpointsWithAngles.push_back(x);
            midpointsWithAngles.push_back(y);
            // add angle of normal to midpointsWithAngles
            midpointsWithAngles.push_back(atan2(n2, n1));
            numGates++;
        } 
        
        if (line.c_str()[0] == 'p') {
            sscanf(line.c_str(), "p %f %f", &x, &y);
            if (track1) {
                insideTrack.push_back(x);
                insideTrack.push_back(y);
            } else {
                outsideTrack.push_back(x);
                outsideTrack.push_back(y);
            }
            if (!started && track1) {
                trackStartLine[0] = x;
                trackStartLine[1] = y;
                started = true;
            } else if (!track1 && !started) {
                trackStartLine[2] = x;
                trackStartLine[3] = y;
                started = true;
            }
        } else if (line.c_str()[0] == 'c') {
            sscanf(line.c_str(), "c %f %f %f", &carX, &carY, &carAngle);
            for (int i = 0; i < numCars; i++) {
                cars[i].x = carX;
                cars[i].y = carY;
                cars[i].angle = carAngle;
                cars[i].speed = 0.0f;
                cars[i].acceleration = 0.0f;
            }
        }
    }
    fileStream.close();

    trackStartNormal = -glm::normalize(glm::vec2(trackStartLine[3] - trackStartLine[1], trackStartLine[0] - trackStartLine[2]));

    if (!training) {
        // Load inside track into vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * insideTrack.size(), insideTrack.data(), GL_STATIC_DRAW);

        // Load outside track into vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * outsideTrack.size(), outsideTrack.data(), GL_STATIC_DRAW);

        // Load start line into vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(trackStartLine), trackStartLine, GL_STATIC_DRAW);
    }

    // Load inside track into compute buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * insideTrack.size(), insideTrack.data(), GL_STATIC_DRAW);

    // Load outside track into compute buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * outsideTrack.size(), outsideTrack.data(), GL_STATIC_DRAW);

    // Load gates into compute buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[7]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * midpointsWithAngles.size(), midpointsWithAngles.data(), GL_STATIC_DRAW);

    loadCars(training);
}

void resetCarFitness(void) {
    // Reset just fitness scores
    for (int i = 0; i < numCars; i++) {
        fitness[i] = 0.0f; // fitness
    }

    // Write back to buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars, &fitness[0], GL_DYNAMIC_DRAW);
}

pair<int, int> decideTrainingTracks(void) {
    int numCW, numCCW;
    ifstream numTracksFile;
    numTracksFile.open("assets/tracks/training/anticlockwise/numTracks.txt");
    numTracksFile >> numCCW;
    numTracksFile.close();
    numTracksFile.open("assets/tracks/training/clockwise/numTracks.txt");
    numTracksFile >> numCW;
    numTracksFile.close();
    int CCWChoice = rand() % numCCW + 1;
    int CWChoice = rand() % numCW + 1;
    return make_pair(CCWChoice, CWChoice);
}

void cycleTracks(bool training) {
    ifstream numTracksFile;
    numTracksFile.open("assets/tracks/numTracks.txt");
    numTracksFile >> numTracks;
    numTracksFile.close();
    curTrack = curTrack % numTracks + 1;
    string trackName = "assets/tracks/track" + to_string(curTrack) + ".tr";

    loadTrack(trackName, training);
}

void setupScene() {
    // Create vaos
    glGenVertexArrays(numVAOs, vao);
    glBindVertexArray(vao[0]);

    // Create vbos
    glGenBuffers(numVBOs, vbo);

    // Create vcbos 
    glGenBuffers(numVCBs, vcbo);
}

void init(void) {
    Utils::setScreenDimensions(windowWidth, windowHeight);
    trackRenderingProgram = Utils::createShaderProgram("shaders/trackVert.glsl", "shaders/trackFrag.glsl");
    carRenderingProgram = Utils::createShaderProgram("shaders/carVert.glsl", "shaders/carFrag.glsl");
    wheelComputeShader = Utils::createShaderProgram("shaders/carPointsCS.glsl");
    physicsComputeShader = Utils::createShaderProgram("shaders/carPhysicsCS.glsl");
    driverRenderingProgram = Utils::createShaderProgram("shaders/driverVert.glsl", "shaders/driverFrag.glsl");
    tsRenderingProgram = Utils::createShaderProgram("shaders/startVert.glsl", "shaders/startFrag.glsl");
    computerVisionComputeShader = Utils::createShaderProgram("shaders/computerVisionCS.glsl");
    computerVisionRenderingProgram = Utils::createShaderProgram("shaders/startVert.glsl", "shaders/startFrag.glsl");
    fittnessComputeShader = Utils::createShaderProgram("shaders/fitnessCS.glsl");
}

void display(GLFWwindow *window) {
    // Clear the screen
    glClear(GL_DEPTH_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    renderComputerVision();

    if (showTrack) {
        glUseProgram(trackRenderingProgram);
        // Draw the inside track
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)(insideTrack.size() / 2));

        // Draw the outside track
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)(outsideTrack.size() / 2));

        // Draw the normals
        TrainingTrackMaker::visualizeNormals(window, &normals, &midpoints);
    }

    // Draw the start line
    glUseProgram(tsRenderingProgram);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_LINE_STRIP, 0, 2);

    // Draw the cars - in future get car points from a compute shader and draw that buffer instead
    glUseProgram(carRenderingProgram);

    colLoc = glGetUniformLocation(carRenderingProgram, "colourIn");

    // Front wheels
    glUniform4f(colLoc, 0.6f, 0.6f, 0.6f, 1.0f);
    glPointSize(6.0f);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 2 * sizeof(float), (void*)(0 * 2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, numCars);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 2 * sizeof(float), (void*)(1 * 2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, numCars);

    // Rear wheels
    glUniform4f(colLoc, 0.9f, 0.9f, 0.9f, 1.0f);
    glPointSize(8.0f);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 2 * sizeof(float), (void*)(2 * 2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, numCars);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 2 * sizeof(float), (void*)(3 * 2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, numCars);

    // Driver
    glUseProgram(driverRenderingProgram);
    glPointSize(5.0f);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 2 *sizeof(float), (void*)(4 * 2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glDrawArrays(GL_POINTS, 0, numCars);
}

void setInput(int offset, float value) {
    inputs[offset] = value;
}

void getPlayerInputs(GLFWwindow *window) {
    for (int i = 0; i < numInputs * numDrivers; i++) {
        if (glfwGetKey(window, carInputs[i]) == GLFW_PRESS) {
            setInput(i, 1.0f);
        } else {
            setInput(i, 0.0f);
        }
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numInputs * numCars, &inputs[0], GL_DYNAMIC_DRAW);
}

void visualiseSimulation(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !cHeld) {
        cycleTracks(false);
        cHeld = true;
    } else if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE) {
        cHeld = false;
    }

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !sHeld) {
        showTrack = !showTrack;
        sHeld = true;
    } else if (glfwGetKey(window, GLFW_KEY_P) == GLFW_RELEASE) {
        sHeld = false;
    }

    calculateCarWheels();

    display(window);
    glfwSwapBuffers(window);
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
        shouldCreateTrack = true;
    }
}

void setupSimulation(bool visual) {
    glGenBuffers(numCBs, cbo);

    // Computer vision angles
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[6]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numComputerVisionAngles, computerVisionAngles, GL_STATIC_DRAW);

    // Initial (0) inputs
    for (int i = 0; i < numInputs * numCars; i++) {
        inputs[i] = 0.0f;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numInputs * numCars, &inputs[0], GL_DYNAMIC_DRAW);

    // Initial cv distances
    for (int i = 0; i < numCars * numComputerVisionAngles; i++) {
        computerVisionDistances[i] = 0.0f;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[5]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numComputerVisionAngles, &computerVisionDistances[0], GL_DYNAMIC_DRAW);

    if (visual) {
        setupScene();
        cycleTracks(false);
    }
}

void runSimulation() {
    calculateCarPhysics();
    calculateFitness();
    calculateComputerVision();
}

void runFrame(GLFWwindow *window, double currentTime, bool training) {
    deltaTime = currentTime - lastTime;
    lastTime = currentTime;

    runSimulation();
    if (!training) {
        getPlayerInputs(window);
        #ifndef DONT_USE_NNS
        DeepNeuralNets::invokeNeuralNets(glm::vec4(trackStartLine[0], trackStartLine[1], trackStartLine[2], trackStartLine[3]));
        #endif
        visualiseSimulation(window);

        // fitness
        // glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[1]);
        // glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars, &fitness[0]);
        // cout << "fitness: " << fitness[0] << endl;
    }
}

// Dont want to teach overconfidence
int framesPerEpoch(int epochs) {
    // if (epochs < 200) {
    //     return 60 * (5 + epochs / 10);
    // }
    return 60 * (25 + rand() % 10);
}

void trainNeuralNets(int epochs, int epochWriteGap) {
    #ifndef DONT_USE_NNS
    for (int i = 1; i < epochs + 1; i++) {
        // Set car fitness to 0 
        resetCarFitness();

        // Get CCW and CW track choices
        pair<int, int> tracks = decideTrainingTracks();

        // Load CCW track
        string trackName = "assets/tracks/training/anticlockwise/" + to_string(tracks.first) + ".tr";
        trackName = "assets/tracks/track2.tr";
        loadTrack(trackName, true);

        // Run simulation on CCW track
        for (int j = 0; j < framesPerEpoch(i); j++) {
            deltaTime = deterministicDt;// + (double)(rand() % 1000) / 100000.0l;
            DeepNeuralNets::invokeNeuralNets(glm::vec4(trackStartLine[0], trackStartLine[1], trackStartLine[2], trackStartLine[3]));
            runSimulation();
            Utils::checkOpenGLError();
        }

        // // Load CW track
        // trackName = "assets/tracks/training/clockwise/" + to_string(tracks.second) + ".tr";
        // loadTrack(trackName, true);

        // // Run simulation on CW track
        // for (int j = 0; j < framesPerEpochPerTrack; j++) {
        //     deltaTime = deterministicDt + (double)(rand() % 1000) / 100000.0l;
        //     runSimulation();
        //     DeepNeuralNets::invokeNeuralNets(glm::vec4(trackStartLine[0], trackStartLine[1], trackStartLine[2], trackStartLine[3]));
        // }

        DeepNeuralNets::evolveNeuralNets();
        if (i % epochWriteGap == 0) {
            DeepNeuralNets::exportBestModel();
        }
    }
    #endif
}

void setupTraining(void) {
    init();
    setupSimulation(false);

    #ifndef DONT_USE_NNS
    DeepNeuralNets::setupTraining(cbo[0], cbo[5], cbo[2], cbo[1]);
    #endif

    trainNeuralNets(100000, 10);
}

int main(void) {
    srand((unsigned int)time(NULL));

    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    if (TRAINING) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hide the window
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        GLFWwindow* window = glfwCreateWindow(1, 1, "", nullptr, nullptr);
        if (!window) {
            throw std::runtime_error("Failed to create GLFW window");
        }
        glfwMakeContextCurrent(window);
        if (glewInit() != GLEW_OK) {
            cout << "Not ok" << endl;
            exit(EXIT_FAILURE);
        }

        setupTraining();

        exit(EXIT_SUCCESS);
    }

    if (numInputs * numDrivers != sizeof(carInputs) / sizeof(int)) {
        cout << "Number of inputs does not match input array size" << endl;
        exit(EXIT_FAILURE);
    }
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "GeneticRacing", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) { 
        exit(EXIT_FAILURE); 
    }
    glfwSwapInterval(1);

    init();
    setupSimulation(true);
    #ifndef DONT_USE_NNS
    DeepNeuralNets::initNeuralNets(cbo[0], cbo[5], cbo[2], cbo[1]);
    DeepNeuralNets::importModel("assets/models/epoch100_best.txt", 0);
    DeepNeuralNets::importModel("assets/models/epoch80_best.txt", 1);
    #endif
    while (!glfwWindowShouldClose(window)) {
        if (shouldCreateTrack) {
            shouldCreateTrack = TrainingTrackMaker::runTrackFrame(window, glfwGetTime());
            if (!shouldCreateTrack) {
                cycleTracks(false);
            }
        } else {
            runFrame(window, glfwGetTime(), TRAINING);
            frames++;
        }
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
}