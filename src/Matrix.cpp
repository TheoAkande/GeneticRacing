#include "Matrix.h"

bool Matrix::initialized = false;
GLuint Matrix::additionShader, Matrix::multiplicationShader, Matrix::transposeShader;
GLuint Matrix::matCBOs[NUM_MATRIX_CBO];

// Private

void Matrix::setupClass(void) {
    if (Matrix::initialized) return;

    // Load the addition shader
    additionShader = Utils::createShaderProgram("shaders/matrix/addition.glsl");

    // Load the multiplication shader
    multiplicationShader = Utils::createShaderProgram("shaders/matrix/multiplication.glsl");

    // Load the transpose shader
    transposeShader = Utils::createShaderProgram("shaders/matrix/transpose.glsl");

    // Generate compute buffer objects
    glGenBuffers(NUM_MATRIX_CBO, matCBOs);

    Matrix::initialized = true;
}

// Public