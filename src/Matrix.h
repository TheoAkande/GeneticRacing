/*
Matrix class that leverages the GPU to perform parallel matrix operations
Note: data is stored in row-major order (stored row by row)
*/

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
        GLuint matCBOs[NUM_MATRIX_CBO];
        bool dirty; // Whether the data in the compute buffer object is different from the data in the vector
        void invokeShader(GLuint shader, Matrix *m, int invokations, int outputBufferSize);   // Invoke a shader on the matrix
        void setup(void);           // Setup the matrix                                    
        void getData(void);         // Get the data from the compute buffer object                                    
        void outputToInput(void);   // Move data from output SSBO to input SSBO
        void invokeScalar(GLuint shader, float val); // Invoke a shader that takes a scalar
        Matrix(GLuint cbo, int rows, int cols);         // Initialize from a compute buffer object

        static bool initialized;
        static GLuint 
            additionShader, multiplicationShader, transposeShader, 
            scalarMultiplicationShader, subtractionShader;
        static void setupClass(void);
    protected:
        void map(GLuint shader); // Map a function onto each item in the matrix
    public:
        Matrix(vector<float> data, int rows, int cols);
        Matrix(int rows, int cols); // Initialize with zeros
        Matrix(int rows, int cols, float val); // Initialize with a value
        Matrix(int size); // Identity matrix

        Matrix& transpose(void);
        Matrix& operator+(Matrix &m);
        Matrix& operator-(Matrix &m);
        Matrix& operator*(Matrix &m);
        Matrix& operator*(float val);
        Matrix& operator/(float val);

        Matrix& operator+=(Matrix &m);
        Matrix& operator-=(Matrix &m);
        Matrix& operator*=(Matrix &m);
        Matrix& operator*=(float val);
        Matrix& operator/=(float val);

        // Note: to access a single element it is more efficient to use () rather than [][]
        float operator()(int row, int col);
        vector<float>& operator[](int index);

        void transposeSelf(void); // Apply transpose in place

        void addRow(vector<float> row);
        void addRow(float val);
        void addRow(void); // Add a row of zeros

        ~Matrix(void);

        void show(void);
};

#endif