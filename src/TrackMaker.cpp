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

int TrackMaker::numInside = 0;

bool TrainingTrackMaker::projecting = false;
vector<float> TrainingTrackMaker::normals;
vector<float> TrainingTrackMaker::midpoints;
float TrainingTrackMaker::normDir = -1.0f;

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
            glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)(numInside));
        } else {
            glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)(numInside));
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
    string localTrackName = "assets/tracks/track" + to_string(numTracks + 1) + ".tr";
    glm::vec2 start = glm::vec2((insideStart.x + outsideStart.x) / 2.0f, (insideStart.y + outsideStart.y) / 2.0f);
    glm::vec2 startNormal = -glm::normalize(glm::vec2(outsideStart.y - insideStart.y, insideStart.x - outsideStart.x));
    float startAngle = atan2(startNormal.y, startNormal.x);

    ofstream trackFile, localFile;
    trackFile.open(trackName);
    localFile.open(localTrackName);

    trackFile << "c " << start.x - startNormal.x * 0.01 << " " << start.y - startNormal.y * 0.01 << " " << startAngle << endl;
    localFile << "c " << start.x - startNormal.x * 0.01 << " " << start.y - startNormal.y * 0.01 << " " << startAngle << endl; 

    for (int i = 0; i < inside.size(); i += 2) {
        trackFile << "p " << inside[i] << " " << inside[i + 1] << endl;
        localFile << "p " << inside[i] << " " << inside[i + 1] << endl;
    }

    trackFile << "-" << endl;
    localFile << "-" << endl;

    for (int i = 0; i < outside.size(); i += 2) {
        trackFile << "p " << outside[i] << " " << outside[i + 1] << endl;
        localFile << "p " << outside[i] << " " << outside[i + 1] << endl;
    }
    trackFile.close();
    localFile.close();

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

    numInside = inside.size() / 2;
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

TrainingTrackMaker::TrainingTrackMaker() {}

void TrainingTrackMaker::darkenInsideProjection(GLFWwindow *window) {
    glUseProgram(startRenderingProgram);
    glBindBuffer(GL_ARRAY_BUFFER, tvbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float), inside.data() + (inside.size() - 4), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)(2));
}

void TrainingTrackMaker::visualizeNormals(GLFWwindow *window) {
    glUseProgram(startRenderingProgram);
    glBindBuffer(GL_ARRAY_BUFFER, tvbo[0]);
    float points[10];

    for (int i = 0; i < normals.size() / 2; i++) {
        float endX = midpoints[i * 2] + normals[i * 2] * 0.05f;
        float endY = midpoints[i * 2 + 1] + normals[i * 2 + 1] * 0.05f;
        float normAngle = atan2(normals[i * 2 + 1], normals[i * 2]);
        float arrowEnd1Angle = normAngle + 3.14159 / 4;
        float arrowEnd2Angle = normAngle - 3.14159 / 4;

        float arrowEnd1X = endX - 0.01f * cos(arrowEnd1Angle);
        float arrowEnd1Y = endY - 0.01f * sin(arrowEnd1Angle);
        float arrowEnd2X = endX - 0.01f * cos(arrowEnd2Angle);
        float arrowEnd2Y = endY - 0.01f * sin(arrowEnd2Angle);

        points[0] = midpoints[i * 2];
        points[1] = midpoints[i * 2 + 1];
        points[2] = endX;
        points[3] = endY;
        points[4] = arrowEnd1X;
        points[5] = arrowEnd1Y;
        points[6] = endX;
        points[7] = endY;
        points[8] = arrowEnd2X;
        points[9] = arrowEnd2Y;


        glBufferData(GL_ARRAY_BUFFER, 10 * sizeof(float), points, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_LINE_STRIP, 0, 5);
    }
}

bool TrainingTrackMaker::runTrackFrame(GLFWwindow *window, double currentTime) {
    if (!trackSetup) {
        initTrack();
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDepthFunc(GL_LEQUAL);

    numInside = outside.size() / 2;
    displayTrack(window);
    visualizeNormals(window);

    double mx, my;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !clickHeld) {
        clickHeld = true;
        glfwGetCursorPos(window, &mx, &my);
        if (!insideStarted) {
            insideStart.x = Utils::pixelToScreenX((int)mx);
            insideStart.y = Utils::pixelToScreenY(windowTHeight - (int)my);
            startLine[0] = insideStart.x;
            startLine[1] = insideStart.y;
            startLine[2] = insideStart.x;
            startLine[3] = insideStart.y;
            insideStarted = true;
        }
        if (projecting) {
            inside.pop_back();
            inside.pop_back();
        } 
        inside.push_back(Utils::pixelToScreenX((int)mx));
        inside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
        projecting = false;
    } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !outsideStarted) {
        glfwGetCursorPos(window, &mx, &my);
        startLine[2] = Utils::pixelToScreenX((int)mx);
        startLine[3] = Utils::pixelToScreenY(windowTHeight - (int)my);
    } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && outsideStarted && !outsideComplete) {
        if (projecting) {
            outside.pop_back();
            outside.pop_back();
        } else {
            projecting = true;
        }
        glfwGetCursorPos(window, &mx, &my);
        outside.push_back(Utils::pixelToScreenX((int)mx));
        outside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
    }  

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE && clickHeld) {
        clickHeld = false;
        glfwGetCursorPos(window, &mx, &my);
        if (!outsideStarted) {
            outsideStart.x = Utils::pixelToScreenX((int)mx);
            outsideStart.y = Utils::pixelToScreenY(windowTHeight - (int)my);
            startLine[2] = outsideStart.x;
            startLine[3] = outsideStart.y;
            outsideStarted = true;
        }
        if (projecting) {
            outside.pop_back();
            outside.pop_back();
        }
        projecting = false;
        outside.push_back(Utils::pixelToScreenX((int)mx));
        outside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));

        glm::vec2 norm = normDir *
            glm::normalize(
                glm::vec2(outside[outside.size() - 1] - inside[outside.size() - 1], 
                inside[outside.size() - 2] - outside[outside.size() - 2])
            );
        normals.push_back(norm.x);
        normals.push_back(norm.y);
        midpoints.push_back((outside[outside.size() - 2] + inside[outside.size() - 2]) / 2.0f);
        midpoints.push_back((outside[outside.size() - 1] + inside[outside.size() - 1]) / 2.0f);
    } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE && !clickHeld && !insideComplete) {
        if (projecting) {
            inside.pop_back();
            inside.pop_back();
        } else {
            projecting = true;
        }
        glfwGetCursorPos(window, &mx, &my);
        inside.push_back(Utils::pixelToScreenX((int)mx));
        inside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
    }

    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS && !enterHeld) {
        enterHeld = true;
        if (projecting) {
            inside.pop_back();
            inside.pop_back();
        }
        if (!insideComplete) {
            insideComplete = true;
            outsideComplete = true;
        }
    } else if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_RELEASE) {
        enterHeld = false;
    }

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        exportTrack();
        return false;
    }

    // Darken inside projection
    if (projecting && outsideStarted && inside.size() > outside.size()) {
        darkenInsideProjection(window);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();

    return true;
}