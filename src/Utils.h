#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <SOIL2/SOIL2.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#ifndef UTILS_H
#define UTILS_H

class Utils
{
private:
	static int screenWidth, screenHeight;

	static std::string readShaderFile(const char *filePath);
	static void printProgramLog(int prog);
	static GLuint prepareShader(int shaderTYPE, const char *shaderPath);
	static int finalizeShaderProgram(GLuint sprogram);

public:
	Utils();
	static float pixelToScreenX(int x);
	static float pixelToScreenY(int y);

	static float pixelsToScreenWidth(int w);
	static float pixelsToScreenHeight(int h);

	static int screenToPixelX(float x);
	static int screenToPixelY(float y);

	static int screenWidthToPixels(float w);
	static int screenHeightToPixels(float h);

	static void setScreenDimensions(int width, int height);

	static bool checkOpenGLError();
	static void printShaderLog(GLuint shader);
	static GLuint createShaderProgram(const char *cs);
	static GLuint createShaderProgram(const char *vs, const char *fs);
	static GLuint loadTexture(const char *texImagePath);
};

#endif