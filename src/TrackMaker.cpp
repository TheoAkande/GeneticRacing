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
bool TrainingTrackMaker::spaceHeld = false;
bool TrainingTrackMaker::tabHeld = false;
bool TrainingTrackMaker::shiftHeld = false;

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
    // Get number of tracks currently created
    ifstream numTracksFile;
    numTracksFile.open("assets/tracks/numTracks.txt");
    int numTracks;
    numTracksFile >> numTracks;
    numTracksFile.close();

    // Create track name
    string trackName = "../../../src/assets/tracks/track" + to_string(numTracks + 1) + ".tr";
    string localTrackName = "assets/tracks/track" + to_string(numTracks + 1) + ".tr";

    // Calculate starting position and angle information
    glm::vec2 start = glm::vec2((insideStart.x + outsideStart.x) / 2.0f, (insideStart.y + outsideStart.y) / 2.0f);
    glm::vec2 startNormal = -glm::normalize(glm::vec2(outsideStart.y - insideStart.y, insideStart.x - outsideStart.x));
    float startAngle = atan2(startNormal.y, startNormal.x);

    // Open local (build) and persistent (src) track files
    ofstream trackFile, localFile;
    trackFile.open(trackName);
    localFile.open(localTrackName);

    // Write start point
    trackFile << "c " << start.x - startNormal.x * 0.02 << " " << start.y - startNormal.y * 0.02 << " " << startAngle << endl;
    localFile << "c " << start.x - startNormal.x * 0.02 << " " << start.y - startNormal.y * 0.02 << " " << startAngle << endl; 

    // Write inside track
    for (int i = 0; i < inside.size(); i += 2) {
        trackFile << "p " << inside[i] << " " << inside[i + 1] << endl;
        localFile << "p " << inside[i] << " " << inside[i + 1] << endl;
    }

    trackFile << "-" << endl;
    localFile << "-" << endl;

    // Write outside track
    for (int i = 0; i < outside.size(); i += 2) {
        trackFile << "p " << outside[i] << " " << outside[i + 1] << endl;
        localFile << "p " << outside[i] << " " << outside[i + 1] << endl;
    }

    trackFile.close();
    localFile.close();

    // Update the number of tracks
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

// Run the track frame. Returns false if we are done creating the track
bool TrackMaker::runTrackFrame(GLFWwindow *window, double currentTime) {
    // If class hasn't been initialized, do so
    if (!trackSetup) {
        initTrack();
    }

    // Call the display function
    numInside = inside.size() / 2;
    displayTrack(window);
    glfwSwapBuffers(window);
    glfwPollEvents();

    // Check if mouse is newly clicked
    double mx, my;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !clickHeld) {
        clickHeld = true;
        glfwGetCursorPos(window, &mx, &my);
        if (!insideComplete) {          // If still creating inside track
            inside.push_back(Utils::pixelToScreenX((int)mx));
            inside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
        } else if (!outsideComplete) {  // If still creating outside track
            outside.push_back(Utils::pixelToScreenX((int)mx));
            outside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
        }
        if (!insideStarted) {   // If starting inside track
            insideStart.x = Utils::pixelToScreenX((int)mx);
            insideStart.y = Utils::pixelToScreenY(windowTHeight - (int)my);
            startLine[0] = insideStart.x;
            startLine[1] = insideStart.y;
            startLine[2] = insideStart.x;
            startLine[3] = insideStart.y;
            insideStarted = true;
        } else if (insideComplete && !outsideStarted) { // If starting outside track
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
        // Complete the inside or whole track
        enterHeld = true;
        if (!insideComplete) {
            insideComplete = true;
        } else if (!outsideComplete) {
            outsideComplete = true;
        }
    } else if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_RELEASE) {
        enterHeld = false;
    }

    // Export the track (save it)
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        exportTrack();
        return false;
    }

    // Clear the track
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        inside.clear();
        outside.clear();
        insideComplete = false;
        outsideComplete = false;
        insideStarted = false;
        outsideStarted = false;
        startLine[0] = startLine[1] = startLine[2] = startLine[3] = 1.1f;
    }

    return true;
}

TrackMaker::TrackMaker() {}

TrainingTrackMaker::TrainingTrackMaker() {}

// Show the projection of the next point on the inside track
void TrainingTrackMaker::showInsideProjection(GLFWwindow *window) {
    glUseProgram(startRenderingProgram);
    glBindBuffer(GL_ARRAY_BUFFER, tvbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float), inside.data() + (inside.size() - 4), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)(2));
}

// Export the track including writing normal data
void TrainingTrackMaker::exportTrack(void) {
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

    // Write start point
    trackFile << "c " << start.x - startNormal.x * 0.02 << " " << start.y - startNormal.y * 0.02 << " " << startAngle << endl;
    localFile << "c " << start.x - startNormal.x * 0.02 << " " << start.y - startNormal.y * 0.02 << " " << startAngle << endl; 

    // Write normals
    for (int i = 0; i < normals.size(); i += 2) {
        trackFile << "n " << midpoints[i] << " " << midpoints[i + 1] << " " << normals[i] << " " << normals[i + 1] << endl;
        localFile << "n " << midpoints[i] << " " << midpoints[i + 1] << " " << normals[i] << " " << normals[i + 1] << endl;
    }

    // Write inside track
    for (int i = 0; i < inside.size(); i += 2) {
        trackFile << "p " << inside[i] << " " << inside[i + 1] << endl;
        localFile << "p " << inside[i] << " " << inside[i + 1] << endl;
    }

    trackFile << "-" << endl;
    localFile << "-" << endl;

    // Write outside track
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

// Visualize the normals of the track
void TrainingTrackMaker::visualizeNormals(GLFWwindow *window, vector<float> *normals, vector<float> *midpoints) {
    // If class hasn't been initialized, do so
    if (!trackSetup) {
        initTrack();
    }
    
    glUseProgram(startRenderingProgram);
    glBindBuffer(GL_ARRAY_BUFFER, tvbo[0]);
    float points[10];

    // Create points for and draw each arrow
    for (int i = 0; i < normals->size() / 2; i++) {
        float endX = midpoints->at(i * 2) + normals->at(i * 2) * 0.05f;
        float endY = midpoints->at(i * 2 + 1) + normals->at(i * 2 + 1) * 0.05f;
        float normAngle = atan2(normals->at(i * 2 + 1), normals->at(i * 2));
        float arrowEnd1Angle = normAngle + 3.14159 / 4;
        float arrowEnd2Angle = normAngle - 3.14159 / 4;

        float arrowEnd1X = endX - 0.01f * cos(arrowEnd1Angle);
        float arrowEnd1Y = endY - 0.01f * sin(arrowEnd1Angle);
        float arrowEnd2X = endX - 0.01f * cos(arrowEnd2Angle);
        float arrowEnd2Y = endY - 0.01f * sin(arrowEnd2Angle);

        points[0] = midpoints->at(i * 2);
        points[1] = midpoints->at(i * 2 + 1);
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

// Run the track frame. Returns false if we are done creating the track
bool TrainingTrackMaker::runTrackFrame(GLFWwindow *window, double currentTime) {
    // If class hasn't been initialized, do so
    if (!trackSetup) {
        initTrack();
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDepthFunc(GL_LEQUAL);

    // Call the display function
    numInside = outside.size() / 2;
    displayTrack(window);
    visualizeNormals(window, &normals, &midpoints);

    double mx, my;
    // Check if mouse is newly clicked
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !clickHeld) {
        clickHeld = true;
        glfwGetCursorPos(window, &mx, &my);
        if (!insideStarted) {       // If still creating inside track
            insideStart.x = Utils::pixelToScreenX((int)mx);
            insideStart.y = Utils::pixelToScreenY(windowTHeight - (int)my);
            startLine[0] = insideStart.x;
            startLine[1] = insideStart.y;
            startLine[2] = insideStart.x;
            startLine[3] = insideStart.y;
            insideStarted = true;
        }
        if (projecting) {        // If currently projecting a point onto the inside track, remove it
            inside.pop_back();
            inside.pop_back();
        } 
        // Create permanent inside track point
        inside.push_back(Utils::pixelToScreenX((int)mx));
        inside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
        projecting = false;
    } else if (
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !outsideStarted
    ) { // If haven't started the outside track yet (only 1 point has been put down)
        glfwGetCursorPos(window, &mx, &my);
        startLine[2] = Utils::pixelToScreenX((int)mx);
        startLine[3] = Utils::pixelToScreenY(windowTHeight - (int)my);
    } else if (
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && outsideStarted && !outsideComplete
    ) { // If mouse is held and outside track is not complete, project a point outside
        if (projecting) {     // If currently projecting a point onto the outside track, remove it
            outside.pop_back();
            outside.pop_back();
        } else {
            projecting = true;
        }
        glfwGetCursorPos(window, &mx, &my);
        // Create projected outside track point
        outside.push_back(Utils::pixelToScreenX((int)mx));
        outside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
    }  

    if (
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE && clickHeld
    ) { // If mouse is newly released
        clickHeld = false;
        glfwGetCursorPos(window, &mx, &my);
        if (!outsideStarted) {
            outsideStart.x = Utils::pixelToScreenX((int)mx);
            outsideStart.y = Utils::pixelToScreenY(windowTHeight - (int)my);
            startLine[2] = outsideStart.x;
            startLine[3] = outsideStart.y;
            outsideStarted = true;
        }
        if (projecting) {   // If projecting a point onto the outside track, remove it
            outside.pop_back();
            outside.pop_back();
        }
        // Create permanent outside track point
        projecting = false;
        outside.push_back(Utils::pixelToScreenX((int)mx));
        outside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));

        // Calculate normal
        glm::vec2 norm = normDir *
            glm::normalize(
                glm::vec2(outside[outside.size() - 1] - inside[outside.size() - 1], 
                inside[outside.size() - 2] - outside[outside.size() - 2])
            );

        // Add normal and midpoint to vectors
        normals.push_back(norm.x);
        normals.push_back(norm.y);
        midpoints.push_back((outside[outside.size() - 2] + inside[outside.size() - 2]) / 2.0f);
        midpoints.push_back((outside[outside.size() - 1] + inside[outside.size() - 1]) / 2.0f);
    } else if (
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE && !clickHeld && !insideComplete
    ) { // If mouse is released and inside track is not complete, project a point inside
        if (projecting) {   // If projecting a point onto the inside track, remove it
            inside.pop_back();
            inside.pop_back();
        } else {
            projecting = true;
        }
        glfwGetCursorPos(window, &mx, &my);
        // Create projected inside track point
        inside.push_back(Utils::pixelToScreenX((int)mx));
        inside.push_back(Utils::pixelToScreenY(windowTHeight - (int)my));
    }

    // Complete track when "enter" pressed
    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS && !enterHeld) {
        enterHeld = true;
        if (projecting) {   // If projecting a point onto the inside track, remove it
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

    // Export the track (save it)
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        exportTrack();
        return false;
    }

    // Clear the track
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        inside.clear();
        outside.clear();
        normals.clear();
        midpoints.clear();
        insideComplete = false;
        outsideComplete = false;
        insideStarted = false;
        outsideStarted = false;
        projecting = false;
        startLine[0] = startLine[1] = startLine[2] = startLine[3] = 1.1f;
    }

    // Show the inside projection
    if (projecting && outsideStarted && inside.size() > outside.size()) {
        showInsideProjection(window);
    }

    // Reverse orientation of previous normal if space
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !spaceHeld && !clickHeld) {
        spaceHeld = true;

        int nSize = normals.size();
        normals[nSize - 2] = -normals[nSize - 2];
        normals[nSize - 1] = -normals[nSize - 1];

        if (inside.size() > outside.size()) {
            inside.pop_back();
            inside.pop_back();
            projecting = false;
        }

        float lastOutsideY = outside.back();
        outside.pop_back();
        float lastOutsideX = outside.back();
        outside.pop_back();

        float lastInsideY = inside.back();
        inside.pop_back();
        float lastInsideX = inside.back();
        inside.pop_back();

        outside.push_back(lastInsideX);
        outside.push_back(lastInsideY);

        inside.push_back(lastOutsideX);
        inside.push_back(lastOutsideY);
    } else if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) {
        spaceHeld = false;
    }

    // Reverse all orientations if tab 
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS && !tabHeld && !clickHeld) {
        tabHeld = true;
        for (int i = 0; i < normals.size(); i++) {
            normals[i] = -normals[i];
        }

        if (inside.size() > outside.size()) {
            inside.pop_back();
            inside.pop_back();
            projecting = false;
        }
        for (int i = 0; i < inside.size(); i += 2) {
            float temp = inside[i];
            inside[i] = outside[i];
            outside[i] = temp;

            temp = inside[i + 1];
            inside[i + 1] = outside[i + 1];
            outside[i + 1] = temp;
        }
    } else if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_RELEASE) {
        tabHeld = false;
    }

    // OpenGL operations
    glfwSwapBuffers(window);
    glfwPollEvents();

    return true;
}