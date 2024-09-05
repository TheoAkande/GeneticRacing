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

using namespace std;

#ifndef TRACKMAKER_H
#define TRACKMAKER_H

#define numTVAOs 1
#define numTVBOs 3

#define windowTWidth 2000
#define windowTHeight 1500

class TrackMaker
{
    private:
        static GLuint tRenderingProgram, startRenderingProgram;
        static GLuint tvao[numTVAOs];
        static GLuint tvbo[numTVBOs];
        static GLuint uLoc;

        static bool trackSetup;

        static bool insideComplete;
        static bool outsideComplete;
        static bool insideStarted;
        static bool outsideStarted;

        static bool clickHeld;
        static bool enterHeld;

        static vector<float> inside;
        static vector<float> outside;

        static glm::vec2 insideStart;
        static glm::vec2 outsideStart;

        static float startLine[4];

        static int numInside;

        static void displayTrack(GLFWwindow *window);
        static void initTrack(void);
        static void exportTrack(void);
    public:
        TrackMaker();
        static bool runTrackFrame(GLFWwindow *window, double currentTime);

    friend class TrainingTrackMaker;
};

class TrainingTrackMaker : public TrackMaker
{
    private:
        static bool projecting;
        static vector<float> normals;
        static vector<float> midpoints;
        static float normDir; 
        static bool spaceHeld, tabHeld, shiftHeld;

        static void exportTrack(void);
        static void darkenInsideProjection(GLFWwindow *window);
        static void visualizeNormals(GLFWwindow *window);
    public :
        TrainingTrackMaker();
        static bool runTrackFrame(GLFWwindow *window, double currentTime);
};

#endif