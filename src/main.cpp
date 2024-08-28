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

// OpenGL definitions
#define numVBOs 3
#define numVAOs 1
#define numCBs 2
#define windowWidth 2000
#define windowHeight 1500
#define numCars 1
#define numCarFloats 3

#define carWidth 0.015f
#define carHeight 0.03f

GLuint vao[numVAOs];
GLuint vbo[numVBOs];
GLuint cbo[numCBs];
GLuint vMatLoc, cwLoc, chLoc, ncfLoc;
GLuint trackRenderingProgram, carRenderingProgram, wheelComputeShader;

float carPos[numCars * numCarFloats];
float carPoints[numCars * 5 * 2];

double deltaTime = 0.0l;
double lastTime = 0.0l;

glm::mat4 viewMat;

const char *track = "assets/tracks/track1.tr";
vector<float> insideTrack;
vector<float> outsideTrack;
float carX, carY, carAngle; // angle 0 = right, 90 = up

struct Car {
    float x, y, angle;
    float speed;
    float acceleration;
    float accelerationAngle;
};

vector<Car> cars;

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
                cars.push_back({carX, carY, carAngle, 0.0f, 0.0f, 0.0f});
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
    glPointSize(5.0f);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, numCars * 5);
}

void runFrame(GLFWwindow *window, double currentTime) {
    deltaTime = currentTime - lastTime;
    lastTime = currentTime;

    // move car / read input

    display(window);
    glfwSwapBuffers(window);
    glfwPollEvents();
}

int main(void) {
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