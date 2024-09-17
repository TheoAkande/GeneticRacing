#include "Matrix.h"

bool Matrix::initialized = false;
GLuint 
    Matrix::additionShader, Matrix::multiplicationShader, Matrix::transposeShader,
    Matrix::scalarMultiplicationShader, Matrix::subtractionShader;

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

    this->dirty = false;
}

void Matrix::outputToInput(void) {
    // Copy the data from the output to the input buffer
    glBindBuffer(GL_COPY_READ_BUFFER, matCBOs[1]);
    glBindBuffer(GL_COPY_WRITE_BUFFER, matCBOs[0]);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(float) * this->data.size());

    // Set this as dirty
    this->dirty = true;
}

void Matrix::invokeScalar(GLuint shader, float val) {
    // Bind the shader
    glUseProgram(shader);

    // Set the uniforms
    GLuint uLoc = glGetUniformLocation(shader, "rows");
    glUniform1i(uLoc, this->rows);
    uLoc = glGetUniformLocation(shader, "cols");
    glUniform1i(uLoc, this->cols);
    uLoc = glGetUniformLocation(shader, "scalar");
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
}

Matrix::Matrix(GLuint cbo, int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data.resize(rows * cols);
    this->dirty = true;

    // Setup the matrix
    this->setup();

    // Setup the data buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->data.size(), NULL, GL_DYNAMIC_COPY);

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

    // Load the scalar multiplication shader
    scalarMultiplicationShader = Utils::createShaderProgram("shaders/matrix/scalarMultiplication.glsl");

    // Load the subtraction shader
    subtractionShader = Utils::createShaderProgram("shaders/matrix/subtraction.glsl");

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

Matrix& Matrix::transpose(void) {
    // Invoke the transpose shader
    this->invokeShader(transposeShader, nullptr, this->rows * this->cols, this->rows * this->cols);

    // Create a new matrix from the output
    return *new Matrix(matCBOs[1], this->cols, this->rows);
}

Matrix& Matrix::operator+(Matrix &m) {
    // Invoke the addition shader
    this->invokeShader(additionShader, &m, this->rows * this->cols, this->rows * this->cols);

    // Create a new matrix from the output
    return *new Matrix(matCBOs[1], this->rows, this->cols);
}

Matrix& Matrix::operator-(Matrix &m) {
    // Invoke the addition shader
    this->invokeShader(subtractionShader, &m, this->rows * this->cols, this->rows * this->cols);

    // Create a new matrix from the output
    return *new Matrix(matCBOs[1], this->rows, this->cols);
}

Matrix& Matrix::operator*(Matrix &m) {
    // Check if the matrices can be multiplied
    assert(this->cols == m.rows);

    // Invoke the multiplication shader
    this->invokeShader(multiplicationShader, &m, this->rows * m.cols, this->rows * m.cols);

    // Create a new matrix from the output
    return *new Matrix(matCBOs[1], this->rows, m.cols);
}

Matrix& Matrix::operator*(float val) {
    // Invoke the shader
    this->invokeScalar(scalarMultiplicationShader, val);

    // Create a new matrix from the output
    return *new Matrix(matCBOs[1], this->rows, this->cols);
}

Matrix& Matrix::operator/(float val) {
    return *this * (1.0f / val);
}

Matrix& Matrix::operator+=(Matrix &m) {
    // Invoke the addition shader
    this->invokeShader(additionShader, &m, this->rows * this->cols, this->rows * this->cols);

    outputToInput();

    return *this;
}

Matrix& Matrix::operator-=(Matrix &m) {
    // Invoke the addition shader
    this->invokeShader(subtractionShader, &m, this->rows * this->cols, this->rows * this->cols);

    outputToInput();

    return *this;
}

Matrix& Matrix::operator*=(Matrix &m) {
    // Check if the matrices can be multiplied
    assert(this->cols == m.rows);

    // Invoke the multiplication shader
    this->invokeShader(multiplicationShader, &m, this->rows * m.cols, this->rows * m.cols);

    // Change dimensions
    this->cols = m.cols;

    // Resize data
    this->data.resize(this->rows * this->cols);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->data.size(), NULL, GL_DYNAMIC_COPY);

    outputToInput();

    return *this;
}

Matrix& Matrix::operator*=(float val) {
    // Invoke the shader
    this->invokeScalar(scalarMultiplicationShader, val);

    outputToInput();

    return *this;
}

Matrix& Matrix::operator/=(float val) {
    return *this *= (1.0f / val);
}

float Matrix::operator()(int row, int col) {
    assert(row < this->rows && col < this->cols);

    // If not dirty, return the value from the vector
    if (!this->dirty) return this->data[row * this->cols + col];

    // Get the data from the compute buffer object
    float val[1];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (col * this->rows + row), sizeof(float), val);

    return val[0];
}

vector<float>& Matrix::operator[](int row) {
    assert(row < this->rows);

    // Create the vector
    vector<float> *v = new vector<float>(this->cols);

    // If not dirty, return the subset of the vector
    if (!this->dirty) {
        for (int i = 0; i < this->cols; i++) {
            (*v)[i] = this->data[row * this->cols + i];
        }
        return *v;
    }

    // Get the data from the compute buffer object
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * row * this->cols, sizeof(float) * this->cols, v->data());

    return *v;
}

void Matrix::addRow(vector<float> row) {
    assert(row.size() == this->cols);

    // If dirty, get the data from the compute buffer object
    this->getData();

    // Add the row to the data
    this->data.insert(this->data.end(), row.begin(), row.end());
    this->rows++;

    // Copy the data from the vector to the compute buffer object
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->matCBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->data.size(), this->data.data(), GL_DYNAMIC_COPY);
}

void Matrix::addRow(float val) {
    vector<float> row(this->cols, val);
    this->addRow(row);
}

void Matrix::addRow(void) {
    this->addRow(0.0f);
}

void Matrix::show(void) {
    // Get the data from the compute buffer object
    this->getData();

    // Print the matrix
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            cout << this->data[i * this->cols + j] << " ";
        }
        cout << endl;
    }
}

Matrix::~Matrix(void) {
    glDeleteBuffers(NUM_MATRIX_CBO, this->matCBOs);
}