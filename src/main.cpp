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

using namespace std;

#define deterministic false
#define deterministicDt 0.015l

// OpenGL definitions
#define numVBOs 3
#define numVAOs 1
#define numCBs 5
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
#define breakingForce 0.2f
#define vMax 1.0f
#define maxTurningRate 1.2f

GLuint vao[numVAOs];
GLuint vbo[numVBOs];
GLuint cbo[numCBs];
GLuint vMatLoc, cwLoc, chLoc, ncfLoc, colLoc;
GLuint trackRenderingProgram, carRenderingProgram, wheelComputeShader, physicsComputeShader;

float inputs[numInputs * numCars];
float carPos[numCars * numCarFloats];
float carPoints[numCars * 5 * 2];

int carInputs[] = {
    GLFW_KEY_UP,
    GLFW_KEY_DOWN,
    GLFW_KEY_LEFT,
    GLFW_KEY_RIGHT,
    GLFW_KEY_SPACE//,
    // GLFW_KEY_W,
    // GLFW_KEY_S,
    // GLFW_KEY_A,
    // GLFW_KEY_D,
    // GLFW_KEY_LEFT_SHIFT,
};

double deltaTime = 0.0l;
double lastTime = 0.0l;

glm::mat4 viewMat;

float appliedForce, totalForce, airResistance;
float appliedTurning, totalTurning;

const char *track = "assets/tracks/track1.tr";
vector<float> insideTrack;
vector<float> outsideTrack;
float carX, carY, carAngle, carSpeed, carAcceleration; // angle 0 = right, 90 = up
GLuint efLoc, bfLoc, mtrLoc, msLoc, cmLoc, dtLoc, niLoc;

struct Car {
    float x, y, angle;
    float speed;
    float acceleration;
};

vector<Car> cars;

void calculateCarPhysics(void) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numCarFloats, &carPos[0], GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numInputs, &inputs[0], GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numCars * numCarFloats, NULL, GL_STATIC_READ);

    glUseProgram(physicsComputeShader);

    ncfLoc = glGetUniformLocation(physicsComputeShader, "numCarFloats");
    glUniform1i(ncfLoc, numCarFloats);
    niLoc = glGetUniformLocation(physicsComputeShader, "numInputs");
    glUniform1i(niLoc, numInputs);

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

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cbo[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cbo[4]);

    glDispatchCompute(numCars, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbo[4]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * numCars * numCarFloats, &carPos[0]);
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

void loadCars(void) {
    for (int i = 0; i < numCars; i++) {
        carPos[i * numCarFloats] = cars[i].x;
        carPos[i * numCarFloats + 1] = cars[i].y;
        carPos[i * numCarFloats + 2] = cars[i].angle;
        carPos[i * numCarFloats + 3] = cars[i].speed;
        carPos[i * numCarFloats + 4] = cars[i].acceleration;
        carPos[i * numCarFloats + 5] = cars[i].angle;
    }

    glGenBuffers(numCBs, cbo);
    calculateCarWheels();
}

void loadTrack(const char *track) {
    // Load track from file
    insideTrack.clear();
    outsideTrack.clear();
    ifstream fileStream(track, ios::in);
    string line = "";    bool track1 = true;
    float x, y;
    while (!fileStream.eof()) {
        getline(fileStream, line);
        if (line.c_str()[0] == '-') {
            track1 = false;
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
        } else if (line.c_str()[0] == 'c') {
            sscanf(line.c_str(), "c %f %f %f", &carX, &carY, &carAngle);
            for (int i = 0; i < numCars; i++) {
                cars.push_back({carX, carY, carAngle, 0.0f, 0.0f});
            }
        }
    }
    fileStream.close();

    // Load inside track into vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * insideTrack.size(), insideTrack.data(), GL_STATIC_DRAW);

    // Load outside track into vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * outsideTrack.size(), outsideTrack.data(), GL_STATIC_DRAW);

    loadCars();
}

void setupScene(const char *curTrack) {
    // Create vaos
    glGenVertexArrays(numVAOs, vao);
    glBindVertexArray(vao[0]);

    // Create vbos
    glGenBuffers(numVBOs, vbo);

    // Load track
    loadTrack(curTrack);
}

void init(void) {
    Utils::setScreenDimensions(windowWidth, windowHeight);
    trackRenderingProgram = Utils::createShaderProgram("shaders/trackVert.glsl", "shaders/trackFrag.glsl");
    carRenderingProgram = Utils::createShaderProgram("shaders/carVert.glsl", "shaders/carFrag.glsl");
    wheelComputeShader = Utils::createShaderProgram("shaders/carPointsCS.glsl");
    physicsComputeShader = Utils::createShaderProgram("shaders/carPhysicsCS.glsl");

    setupScene(track);
}

void display(GLFWwindow *window) {
    // Clear the screen
    glClear(GL_DEPTH_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(trackRenderingProgram);

    // Set the view matrix
    viewMat = glm::mat4(1.0f);
    vMatLoc = glGetUniformLocation(trackRenderingProgram, "vMatrix");
    glUniformMatrix4fv(vMatLoc, 1, GL_FALSE, glm::value_ptr(viewMat));

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
    glUniform4f(colLoc, 0.0f, 1.0f, 0.0f, 1.0f);
    glPointSize(5.0f);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 2 *sizeof(float), (void*)(4 * 2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, numCars);
}

void setInput(int offset, float value) {
    for (int i = 0; i < numCars; i++) {
        inputs[i * numInputs + offset] = value;
    }
}

void runFrame(GLFWwindow *window, double currentTime) {
    deltaTime = currentTime - lastTime;
    lastTime = currentTime;

    for (int i = 0; i < numInputs; i++) {
        if (glfwGetKey(window, carInputs[i]) == GLFW_PRESS) {
            setInput(i, 1.0f);
        } else {
            setInput(i, 0.0f);
        }
    }
    
    calculateCarPhysics();
    calculateCarWheels();

    display(window);
    glfwSwapBuffers(window);
    glfwPollEvents();
}

int main(void) {
    if (numInputs != sizeof(carInputs) / sizeof(int)) {
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
        runFrame(window, glfwGetTime());
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
}