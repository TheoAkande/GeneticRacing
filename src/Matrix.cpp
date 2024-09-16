#include "Matrix.h"

bool Matrix::initialized = false;
GLuint Matrix::additionShader, Matrix::multiplicationShader, Matrix::transposeShader;
GLuint Matrix::matCBOs[NUM_MATRIX_CBO];

// Private

void Matrix::invokeShader(GLuint shader, Matrix *m, int invokations, int outputBufferSize) {
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

    // Setup the output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCBOs[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * outputBufferSize, NULL, GL_DYNAMIC_COPY);

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

void Matrix::setup(void) {
    Matrix::setupClass();
    // Generate compute buffer objects
    glGenBuffers(NUM_MATRIX_CBO, this->matCBOs);
}

void Matrix::getData(void) {
    if (!this->dirty) return;

    // Copy the data from the compute buffer object to the vector
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * this->data.size(), this->data.data());
}

Matrix::Matrix(GLuint cbo, int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data.resize(rows * cols);
    this->dirty = true;

    // Setup the matrix
    this->setup();

    // Copy the data from the compute buffer to the new one
    glBindBuffer(GL_COPY_READ_BUFFER, cbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, this->matCBOs[0]);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(float) * this->data.size());
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

Matrix::Matrix(vector<float> data, int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = data;
    this->dirty = false;

    // Setup the matrix
    this->setup();

    // Copy the data from the vector to the compute buffer object
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->data.size(), this->data.data(), GL_DYNAMIC_COPY);
}

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data.resize(rows * cols);
    this->dirty = false;

    // Setup the matrix
    this->setup();

    // Copy the data from the vector to the compute buffer object
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->data.size(), this->data.data(), GL_DYNAMIC_COPY);
}

Matrix::Matrix(int rows, int cols, float val) {
    this->rows = rows;
    this->cols = cols;
    this->data.resize(rows * cols, val);
    this->dirty = false;

    // Setup the matrix
    this->setup();

    // Copy the data from the vector to the compute buffer object
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->data.size(), this->data.data(), GL_DYNAMIC_COPY);
}

Matrix::Matrix(int size) {
    this->rows = size;
    this->cols = size;
    this->data.resize(size * size, 0.0f);
    for (int i = 0; i < size; i++) {
        this->data[i * size + i] = 1.0f;
    }
    this->dirty = false;

    // Setup the matrix
    this->setup();

    // Copy the data from the vector to the compute buffer object
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->data.size(), this->data.data(), GL_DYNAMIC_COPY);
}

Matrix Matrix::transpose(void) {
    // Invoke the transpose shader
    this->invokeShader(transposeShader, nullptr, this->rows * this->cols, this->rows * this->cols);

    // Create a new matrix from the output
    return Matrix(matCBOs[1], this->cols, this->rows);
}

Matrix Matrix::operator+(Matrix &m) {
    // Invoke the addition shader
    this->invokeShader(additionShader, &m, this->rows * this->cols, this->rows * this->cols);

    // Create a new matrix from the output
    return Matrix(matCBOs[1], this->rows, this->cols);
}

Matrix Matrix::operator-(Matrix &m) {
    // Negate the matrix
    m *= -1.0f;

    // Invoke the addition shader
    this->invokeShader(additionShader, &m, this->rows * this->cols, this->rows * this->cols);

    // Negate the matrix back
    m *= -1.0f;

    // Create a new matrix from the output
    return Matrix(matCBOs[1], this->rows, this->cols);
}

Matrix Matrix::operator*(Matrix &m) {
    // Check if the matrices can be multiplied
    assert(this->cols == m.rows);

    // Invoke the multiplication shader
    this->invokeShader(multiplicationShader, &m, this->rows * m.cols, this->rows * m.cols);

    // Create a new matrix from the output
    return Matrix(matCBOs[1], this->rows, m.cols);
}

Matrix Matrix::operator*(float val) {
    // Bind the shader
    glUseProgram(scalarMultiplicationShader);

    // Set the uniforms
    GLuint uLoc = glGetUniformLocation(scalarMultiplicationShader, "rows");
    glUniform1i(uLoc, this->rows);
    uLoc = glGetUniformLocation(scalarMultiplicationShader, "cols");
    glUniform1i(uLoc, this->cols);
    uLoc = glGetUniformLocation(scalarMultiplicationShader, "scalar");
    glUniform1f(uLoc, val);

    // Setup the output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCBOs[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->data.size(), NULL, GL_DYNAMIC_COPY);

    // Bind the compute buffer objects
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matCBOs[0]);  // Input A
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matCBOs[1]);  // Output

    // Dispatch the compute shader
    glDispatchCompute(this->rows * this->cols, 1, 1);

    // Wait for the shader to finish
    glFinish();

    // Create a new matrix from the output
    return Matrix(matCBOs[1], this->rows, this->cols);
}