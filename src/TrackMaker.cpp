#include "Utils.h"
#include "TrackMaker.h"

using namespace std;

GLuint TrackMaker::tRenderingProgram, TrackMaker::startRenderingProgram;
GLuint TrackMaker::tvao[numTVAOs];
GLuint TrackMaker::tvbo[numTVBOs];
GLuint TrackMaker::uLoc;

bool TrackMaker::trackSetup = false;

bool TrackMaker::insideComplete = false;
bool TrackMaker::outsideComplete = false;
bool TrackMaker::insideStarted = false;
bool TrackMaker::outsideStarted = false;

bool TrackMaker::clickHeld = false;
bool TrackMaker::enterHeld = false;

vector<float> TrackMaker::inside;
vector<float> TrackMaker::outside;

glm::vec2 TrackMaker::insideStart;
glm::vec2 TrackMaker::outsideStart;

float TrackMaker::startLine[4];

void TrackMaker::displayTrack(GLFWwindow *window) {
    // Clear the screen
    glClear(GL_DEPTH_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(tRenderingProgram);

    // Draw the inside track
    if (insideStarted) {
        glBindBuffer(GL_ARRAY_BUFFER, tvbo[0]);
        glBufferData(GL_ARRAY_BUFFER, inside.size() * sizeof(float), inside.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        if (insideComplete) {
            glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)(inside.size() / 2));
        } else {
            glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)(inside.size() / 2));
        }
    }

    if (outsideStarted) {
        // Draw the outside track
        glBindBuffer(GL_ARRAY_BUFFER, tvbo[1]);
        glBufferData(GL_ARRAY_BUFFER, outside.size() * sizeof(float), outside.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        if (outsideComplete) {
            glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)(outside.size() / 2));
        } else {
            glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)(outside.size() / 2));
        }
    }

    // Draw the start line
    if (insideStarted) {
        glUseProgram(startRenderingProgram);
        glBindBuffer(GL_ARRAY_BUFFER, tvbo[2]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(startLine), startLine, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_LINES, 0, 2);
    }
    
}

void TrackMaker::initTrack(void) {
    tRenderingProgram = Utils::createShaderProgram("shaders/trackVert.glsl", "shaders/trackFrag.glsl");
    startRenderingProgram = Utils::createShaderProgram("shaders/startVert.glsl", "shaders/startFrag.glsl");
    // Create vaos
    glGenVertexArrays(numTVAOs, tvao);
    glBindVertexArray(tvao[0]);

    // Create vbos
    glGenBuffers(numTVBOs, tvbo);

    startLine[0] = startLine[1] = startLine[2] = startLine[3] = 1.1f;
    trackSetup = true;
}

void TrackMaker::exportTrack(void) {
    ifstream numTracksFile;
    numTracksFile.open("assets/tracks/numTracks.txt");
    int numTracks;
    numTracksFile >> numTracks;
    numTracksFile.close();

    string trackName = "../../../src/assets/tracks/track" + to_string(numTracks + 1) + ".tr";
    glm::vec2 start = glm::vec2((insideStart.x + outsideStart.x) / 2.0f, (insideStart.y + outsideStart.y) / 2.0f);
    glm::vec2 startNormal = glm::normalize(glm::vec2(outsideStart.y - insideStart.y, insideStart.x - outsideStart.x));
    float startAngle = atan2(-startNormal.y, -startNormal.x);

    ofstream trackFile;
    trackFile.open(trackName);

    trackFile << "c " << start.x - startNormal.x * 0.01 << " " << start.y - startNormal.y * 0.01 << " " << startAngle << endl; 

    for (int i = 0; i < inside.size(); i += 2) {
        trackFile << "p " << inside[i] << " " << inside[i + 1] << endl;
    }

    trackFile << "-" << endl;

    for (int i = 0; i < outside.size(); i += 2) {
        trackFile << "p " << outside[i] << " " << outside[i + 1] << endl;
    }
    trackFile.close();

    ofstream numTracksFileOut;
    numTracksFileOut.open("assets/tracks/numTracks.txt");
    numTracksFileOut << numTracks + 1;
    numTracksFileOut.close();

    ofstream perisitentTracksFileOut;
    perisitentTracksFileOut.open("../../../src/assets/tracks/numTracks.txt");
    if (perisitentTracksFileOut.is_open()) {
        perisitentTracksFileOut << numTracks + 1;
    } else {
        cout << "Unable to open" << endl;
    }
    perisitentTracksFileOut.close();
}

bool TrackMaker::runTrackFrame(GLFWwindow *window, double currentTime) {
    if (!trackSetup) {
        initTrack();
    }

    displayTrack(window);
    glfwSwapBuffers(window);
    glfwPollEvents();

    double mx, my;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !clickHeld) {
        clickHeld = true;
        glfwGetCursorPos(window, &mx, &my);
        if (!insideComplete) {
            inside.push_back(Utils::pixelToScreenX((int)mx));
            inside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
        } else if (!outsideComplete) {
            outside.push_back(Utils::pixelToScreenX((int)mx));
            outside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
        }
        if (!insideStarted) {
            insideStart.x = Utils::pixelToScreenX((int)mx);
            insideStart.y = Utils::pixelToScreenY(windowTHeight - (int)my);
            startLine[0] = insideStart.x;
            startLine[1] = insideStart.y;
            startLine[2] = insideStart.x;
            startLine[3] = insideStart.y;
            insideStarted = true;
        } else if (insideComplete && !outsideStarted) {
            outsideStart.x = Utils::pixelToScreenX((int)mx);
            outsideStart.y = Utils::pixelToScreenY(windowTHeight - (int)my);
            startLine[2] = outsideStart.x;
            startLine[3] = outsideStart.y;
            outsideStarted = true;
        }
    } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
        clickHeld = false;
    }
    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS && !enterHeld) {
        enterHeld = true;
        if (!insideComplete) {
            insideComplete = true;
        } else if (!outsideComplete) {
            outsideComplete = true;
        }
    } else if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_RELEASE) {
        enterHeld = false;
    }

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        exportTrack();
        return false;
    }

    return true;
}

TrackMaker::TrackMaker() {}
