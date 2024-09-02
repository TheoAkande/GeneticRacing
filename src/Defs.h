#ifndef DEFS_H
#define DEFS_H

#define deterministic false
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
#define numCars 100
#define numDrivers 0
#define numCarFloats 5
/*
    0: x
    1: y
    2: angle
    3: speed
    4: acceleration
*/
#define numCarFitnessFloats 6
/*
    0: x _interval_ ago
    1: y _interval_ ago
    2: distance travelled
    3: total speed
    4: laps
    5: fitness
*/
#define numCBs 7
/*
    0: carPos
    1: carFitness
    2: inputs
    3: insideTrack
    4: outsideTrack
    5: computerVisionDistances
    6: computerVisionAngles
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
#define MAX_SSBO 16

#define TRAINING true

#define basePath "../../../"

#endif