#ifndef DEFS_H
#define DEFS_H

#define TRAINING false
#define DONT_USE_NNS // Comment out if using neural networks

#define deterministic true
#define deterministicDt 0.015l

// OpenGL definitions
#define numVBOs 6
/*
    0: insideTrack
    1: outsideTrack
    2: carPoints
    3: driverColours
    4: startLine
    5: computerVisionPoints
*/
#define numVAOs 1
#define windowWidth 2000
#define windowHeight 1500
#define numCars (TRAINING ? 1000 : 2)
#define numDrivers (TRAINING ? 0 : 2)
#define numCarFloats 8
/*
    0: x
    1: y
    2: angle
    3: speed
    4: acceleration
    5: next gate x
    6: next gate y
    7: next gate angle
*/
#define numCarEvalFloats 3
/*
    0: next gate (number) to be passed
    1: total number of gates passed
    2: total speed (to get average, divide by seconds)
*/
#define numCBs 9
/*
    0: carPos
    1: carFitness
    2: inputs
    3: insideTrack
    4: outsideTrack
    5: computerVisionDistances
    6: computerVisionAngles
    7: gates                    - the midpoints and angles of each gate
    8: carEvalData              - data to evaluate car fitness
*/
#define numVCBs 1
/*
    0: carWheelPoints
*/

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
#define MAX_SSBO 96 // we can have 96 total SSBOs
#define MAX_SSBO_IN_USE 16 // we can pass 16 ssbos max to a compute shader

#define basePath "../../../"

#endif