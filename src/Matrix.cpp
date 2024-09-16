#include "Matrix.h"

bool Matrix::initialized = false;
GLuint Matrix::additionShader, Matrix::multiplicationShader, Matrix::transposeShader;
GLuint Matrix::matCBOs[NUM_MATRIX_CBO];

// Private

void Matrix::invokeShader(GLuint shader, Matrix *m, int invokations) {
    // Bind the shader
    glUseProgram(shader);

    // Check if it is a unary operation
    bool unary = m == nullptr;

    // Set the uniforms
    GLuint uLoc = glGetUniformLocation(shader, "rowsA");
    glUniform1i(uLoc, this->rows);
    uLoc = glGetUniformLocation(shader, "colsA");
    glUniform1i(uLoc, this->cols);

    if (!unary) {
        uLoc = glGetUniformLocation(shader, "rowsB");
        glUniform1i(uLoc, m->rows);
        uLoc = glGetUniformLocation(shader, "colsB");
        glUniform1i(uLoc, m->cols);
    }

    // Bind the compute buffer objects
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matCBOs[0]);  // Input A
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matCBOs[1]);  // Output
    if (!unary) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m->matCBOs[0]);  // Input B
    }

    // Dispatch the compute shader
    glDispatchCompute(invokations, 1, 1);

    // Wait for the shader to finish
    glFinish();
}

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