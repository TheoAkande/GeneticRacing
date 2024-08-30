#ifndef DEFS_H
#define DEFS_H

#define deterministic false
#define deterministicDt 0.015l

// OpenGL definitions
#define numVBOs 6
#define numVAOs 1
#define numCBs 12
#define windowWidth 2000
#define windowHeight 1500
#define numCars 2
#define numCarFloats 9

// Car definitions
#define carWidth 0.02f
#define carHeight 0.03f

#define numInputs 5

// Physics definitions
#define frictionMax 0.05f
#define carMass 1.0f
#define carForce 0.1f
#define breakingForce 0.3f
#define vMax 1.0f
#define maxTurningRate 1.2f

#define numComputerVisionAngles 15

#define calculateDistanceInterval 60

#define MAX_FLOATS 268435456
#define MAX_SSBO_SIZE 1073741824
#define MAX_SSBO 16

#endif