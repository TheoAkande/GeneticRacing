// TrackMaker is the framework to create tracks for the cars to be trained on and drive.
// The original TrackMaker class creates the inside track and then the outside track, allowing
// for different numbers of points to be used on the inside and outside track.
// The TrainingTrackMaker class is a subclass of TrackMaker that requires the inside and outside 
// tracks to have the same number of points. It then generates normals and midpoints intended
// to help the neural network learn to drive on the track.
// Note: I never intend to feed the entire track into the neural net, or fitness function, but
//       rather feed the computer certain computer vision data.

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
    public :
        TrainingTrackMaker();
        static void visualizeNormals(GLFWwindow *window, vector<float> *normals, vector<float> *midpoints);
        static bool runTrackFrame(GLFWwindow *window, double currentTime);
};

#endif