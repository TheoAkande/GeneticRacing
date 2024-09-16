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
#include <cstdlib>
#include <chrono>
#include <time.h>

#include "Utils.h"

using namespace std;

#ifndef MATRIX_H
#define MATRIX_H

#define NUM_MATRIX_CBO 2
/*
    0: input A
    1: output
    note: input B can come from the other matrix
*/

class Matrix
{
    private:
        vector<float> data;
        int rows, cols;
        bool dirty; // Whether the data in the compute buffer object is different from the data in the vector
        void invokeShader(GLuint shader, Matrix *m, int invokations, int outputBufferSize);   // Invoke a shader on the matrix
        void setup(void);                                               // Setup the matrix
        void getData(void);                                             // Get the data from the compute buffer object
        Matrix(GLuint cbo, int rows, int cols);         // Initialize from a compute buffer object

        static bool initialized;
        static GLuint additionShader, multiplicationShader, transposeShader;
        static GLuint matCBOs[NUM_MATRIX_CBO];
        static void setupClass(void);
    public:
        Matrix(vector<float> data, int rows, int cols);
        Matrix(int rows, int cols); // Initialize with zeros
        Matrix(int rows, int cols, float val); // Initialize with a value
        Matrix(int size); // Identity matrix

        Matrix transpose(void);
        Matrix operator+(Matrix &m);
        Matrix operator-(Matrix &m);
        Matrix operator*(Matrix &m);
        Matrix operator*(float val);
        Matrix operator/(float val);

        Matrix operator+=(Matrix &m);
        Matrix operator-=(Matrix &m);
        Matrix operator*=(Matrix &m);
        Matrix operator*=(float val);
        Matrix operator/=(float val);

        void show(void);
        void destroy(void);
};

#endif