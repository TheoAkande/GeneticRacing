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

#include "Utils.h"
#include "TrackMaker.h"

using namespace std;

#define deterministic false
#define deterministicDt 0.015l

// OpenGL definitions
#define numVBOs 6
#define numVAOs 1
#define numCBs 10
#define windowWidth 2000
#define windowHeight 1500
#define numCars 2
#define numCarFloats 6

// Car definitions
#define carWidth 0.02f
#define carHeight 0.03f

#define numInputs 5

// Physics definitions
#define frictionMax 0.05f
#define carMass 1.0f
#define carForce 0.1f
#define breakingForce 0.3f
#define vMax 1.0f
#define maxTurningRate 1.2f

#define numComputerVisionAngles 15

GLuint vao[numVAOs];
GLuint vbo[numVBOs];
GLuint cbo[numCBs];
GLuint cwLoc, chLoc, ncfLoc, colLoc;
GLuint trackRenderingProgram, carRenderingProgram, wheelComputeShader, 
    physicsComputeShader, driverRenderingProgram, tsRenderingProgram,
    computerVisionComputeShader, computerVisionRenderingProgram;

float inputs[numInputs * numCars];
float carPos[numCars * numCarFloats];
float carPoints[numCars * 5 * 2];

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
    1.0f, 1.0f, 0.0f, 1.0f
};

double deltaTime = 0.0l;
double lastTime = 0.0l;

glm::mat4 viewMat;

float appliedForce, totalForce, airResistance;
float appliedTurning, totalTurning;

const char *track = "assets/tracks/track6.tr";
vector<float> insideTrack;
vector<float> outsideTrack;
float trackStartLine[4];
glm::vec2 trackStartNormal;
float carX, carY, carAngle, carSpeed, carAcceleration; // angle 0 = right, 90 = up
GLuint efLoc, bfLoc, mtrLoc, msLoc, cmLoc, dtLoc, niLoc, nt1Loc, nt2Loc;
float active[numCars];

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

void renderComputerVision(void) {
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
    // Car data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numCarFloats, &carPos[0], GL_STATIC_DRAW);

    // Inside Track
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[5]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * insideTrack.size(), insideTrack.data(), GL_STATIC_DRAW);

    // Outside Track
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[6]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * outsideTrack.size(), outsideTrack.data(), GL_STATIC_DRAW);

    // Computer Vision Angles
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[8]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numComputerVisionAngles, computerVisionAngles, GL_STATIC_DRAW);

    // Computer Vision Distances
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[9]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numComputerVisionAngles, NULL, GL_STATIC_READ);

    glUseProgram(computerVisionComputeShader);

    ncfLoc = glGetUniformLocation(computerVisionComputeShader, "numCarFloats");
    glUniform1i(ncfLoc, numCarFloats);
    ncfLoc = glGetUniformLocation(computerVisionComputeShader, "numComputerVisionAngles");
    glUniform1i(ncfLoc, numComputerVisionAngles);
    nt1Loc = glGetUniformLocation(computerVisionComputeShader, "numInsidePoints");
    glUniform1i(nt1Loc, insideTrack.size() / 2);
    nt2Loc = glGetUniformLocation(computerVisionComputeShader, "numOutsidePoints");
    glUniform1i(nt2Loc, outsideTrack.size() / 2);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cbo[5]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cbo[6]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo[8]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cbo[9]);

    glDispatchCompute(numCars, numComputerVisionAngles, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[9]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars * numComputerVisionAngles, &computerVisionDistances[0]);
}

void calculateCarPhysics(void) {
    // Original car data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numCarFloats, &carPos[0], GL_STATIC_DRAW);

    // Inputs
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numInputs, &inputs[0], GL_STATIC_DRAW);

    // New car data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numCarFloats, NULL, GL_STATIC_READ);

    // Inside Track
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[5]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * insideTrack.size(), insideTrack.data(), GL_STATIC_DRAW);

    // Outside Track
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[6]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * outsideTrack.size(), outsideTrack.data(), GL_STATIC_DRAW);

    // Car active
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[7]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars, NULL, GL_STATIC_READ);

    glUseProgram(physicsComputeShader);

    ncfLoc = glGetUniformLocation(physicsComputeShader, "numCarFloats");
    glUniform1i(ncfLoc, numCarFloats);
    niLoc = glGetUniformLocation(physicsComputeShader, "numInputs");
    glUniform1i(niLoc, numInputs);
    nt1Loc = glGetUniformLocation(physicsComputeShader, "numInsideTrackPoints");
    glUniform1i(nt1Loc, insideTrack.size() / 2);
    nt2Loc = glGetUniformLocation(physicsComputeShader, "numOutsideTrackPoints");
    glUniform1i(nt2Loc, outsideTrack.size() / 2);

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

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cbo[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cbo[4]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo[5]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cbo[6]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, cbo[7]);

    glDispatchCompute(numCars, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[4]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars * numCarFloats, &carPos[0]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[7]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars, &active[0]);
}

void calculateCarWheels(void) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numCarFloats, &carPos[0], GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * 5 * 2, NULL, GL_STATIC_READ);

    glUseProgram(wheelComputeShader);

    ncfLoc = glGetUniformLocation(wheelComputeShader, "numCarFloats");
    glUniform1i(ncfLoc, numCarFloats);
    cwLoc = glGetUniformLocation(wheelComputeShader, "carWidth");
    glUniform1f(cwLoc, carWidth);
    chLoc = glGetUniformLocation(wheelComputeShader, "carHeight");
    glUniform1f(chLoc, carHeight);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cbo[1]);

    glDispatchCompute(numCars, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[1]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars * 5 * 2, &carPoints[0]);

    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * numCars * 5 * 2, &carPoints[0], GL_STATIC_DRAW);
}

void loadCars(bool createBuffers) {
    for (int i = 0; i < numCars; i++) {
        carPos[i * numCarFloats] = cars[i].x;
        carPos[i * numCarFloats + 1] = cars[i].y;
        carPos[i * numCarFloats + 2] = cars[i].angle;
        carPos[i * numCarFloats + 3] = cars[i].speed;
        carPos[i * numCarFloats + 4] = cars[i].acceleration;
        carPos[i * numCarFloats + 5] = 1.0f;

        active[i] = 1.0f;
    }

    if (createBuffers) {
        glGenBuffers(numCBs, cbo);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbo[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(carColours), &carColours[0], GL_STATIC_DRAW);

    calculateCarWheels();
}

void loadTrack(string track, bool createBuffers = true) {
    // Load track from file
    insideTrack.clear();
    outsideTrack.clear();
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

    // Load inside track into vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * insideTrack.size(), insideTrack.data(), GL_STATIC_DRAW);

    // Load outside track into vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * outsideTrack.size(), outsideTrack.data(), GL_STATIC_DRAW);

    // Load start line into vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(trackStartLine), trackStartLine, GL_STATIC_DRAW);

    loadCars(createBuffers);
}

void cycleTracks(bool createBuffers) {
    ifstream numTracksFile;
    numTracksFile.open("assets/tracks/numTracks.txt");
    numTracksFile >> numTracks;
    numTracksFile.close();
    curTrack = curTrack % numTracks + 1;
    string trackName = "assets/tracks/track" + to_string(curTrack) + ".tr";

    loadTrack(trackName, createBuffers);
}

void setupScene(const char *curTrack) {
    // Create vaos
    glGenVertexArrays(numVAOs, vao);
    glBindVertexArray(vao[0]);

    // Create vbos
    glGenBuffers(numVBOs, vbo);
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

    setupScene(track);
    cycleTracks(true);
}

void display(GLFWwindow *window) {
    // Clear the screen
    glClear(GL_DEPTH_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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

    // Computer Vision
    renderComputerVision();
}

void setInput(int offset, float value) {
    inputs[offset] = value;
}

void determineCarIntersects(void) {
    for (int i = 0; i < numCars; i++) {
        carPos[i * numCarFloats + 5] = active[i];
        if (active[i] == 0.0f) {
            carPos[i * numCarFloats] = 0.0f;
            carPos[i * numCarFloats + 1] = 0.0f;
            carPos[i * numCarFloats + 2] = 0.0f;
            carPos[i * numCarFloats + 3] = 0.0f;
            carPos[i * numCarFloats + 4] = 0.0f;
        } else if (active[i] == -1.0f) {
            cout << "Car " << i << " has completed a lap" << endl;
        }
    }
}

void runFrame(GLFWwindow *window, double currentTime) {
    deltaTime = currentTime - lastTime;
    lastTime = currentTime;

    for (int i = 0; i < numInputs * numCars; i++) {
        if (glfwGetKey(window, carInputs[i]) == GLFW_PRESS) {
            setInput(i, 1.0f);
        } else {
            setInput(i, 0.0f);
        }
    }

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
    
    calculateCarPhysics();

    // computer vision
    calculateComputerVision();

    determineCarIntersects();
    calculateCarWheels();

    display(window);
    glfwSwapBuffers(window);
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
        shouldCreateTrack = true;
    }
}

int main(void) {
    if (numInputs * numCars != sizeof(carInputs) / sizeof(int)) {
        cout << "Number of inputs does not match input array size" << endl;
        exit(EXIT_FAILURE);
    }

    if (!glfwInit()) { 
        exit(EXIT_FAILURE); 
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "GeneticRacing", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) { 
        exit(EXIT_FAILURE); 
    }
    glfwSwapInterval(1);
    init();
    while (!glfwWindowShouldClose(window)) {
        if (shouldCreateTrack) {
            shouldCreateTrack = TrackMaker::runTrackFrame(window, glfwGetTime());
            if (!shouldCreateTrack) {
                cycleTracks(false);
            }
        } else {
            runFrame(window, glfwGetTime());
        }
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
}