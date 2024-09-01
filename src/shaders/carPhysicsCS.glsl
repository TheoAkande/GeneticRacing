#version 430

layout(local_size_x = 1) in;

/*
    Car data:
    0: x
    1: y
    2: angle
    3: speed
    4: acceleration
    5: active
    6: past x
    7: past y
    8: total distance
    9: total speed

    Car inputs:
    0: forward
    1: backward
    2: left
    3: right
    4: brake
*/

layout(binding = 0) buffer buffer1 { float carData[]; };
layout(binding = 1) buffer buffer2 { float carInputs[]; };
layout(binding = 2) buffer buffer3 {float insideTrack[]; };
layout(binding = 3) buffer buffer4 {float outsideTrack[]; };
layout(binding = 4) buffer buffer5 {float carFitness[]; };

uniform int numCarFloats;
uniform int numInputs;
uniform int numFitnessFloats;
uniform int numInsideTrackPoints;
uniform int numOutsideTrackPoints;

uniform vec2 insideStart;
uniform vec2 outsideStart;
uniform vec2 startNormal;
uniform vec2 startPoint;
uniform float startAngle;

uniform float engineForce;
uniform float brakeForce;
uniform float maxTurnRate;
uniform float maxSpeed;
uniform float carMass;
uniform float deltaTime;

uniform int calNewDistance;

struct Line {
    vec2 p1;
    vec2 p2;
};

bool almostEqual(float a, float b) {
    float epsilon = 0.000001;
    return abs(a - b) <= epsilon;
}

float fmax(float a, float b) {
    return a > b ? a : b;
}

float fmin(float a, float b) {
    return a < b ? a : b;
}

// 0 -> collinear
// 1 -> clockwise
// 2 -> counterclockwise
int orientation(vec2 p, vec2 q, vec2 r) {
    float val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (almostEqual(val, 0.0)) return 0;
    return (val > 0) ? 1 : 2;
}

bool on_segment(vec2 p, vec2 q, vec2 r) {
    return q.x <= fmax(p.x, r.x) && q.x >= fmin(p.x, r.x) && q.y <= fmax(p.y, r.y) && q.y >= fmin(p.y, r.y);
}

vec2 order(float n1, float n2) {
    if (n1 < n2) {
        return vec2(n1, n2);
    } else {
        return vec2(n2, n1);
    }
}

// Check if bounding boxes of line and particle intersect
bool boundingIntersect(Line l1, Line l2) {
    vec2 ordered_1 = order(l1.p1.x, l1.p2.x);
    vec2 ordered_2 = order(l2.p1.x, l2.p2.x);
    vec2 ordered_3 = order(l1.p1.y, l1.p2.y);
    vec2 ordered_4 = order(l2.p1.y, l2.p2.y);

    return ordered_1.x <= ordered_2.y && ordered_1.y >= ordered_2.x && ordered_3.x <= ordered_4.y && ordered_3.y >= ordered_4.x;
}

bool intersect(Line l1, Line l2) {
    if (!boundingIntersect(l1, l2)) {   // easily prune most cases
        return false;
    }
    int o1 = orientation(l1.p1, l1.p2, l2.p1);
    int o2 = orientation(l1.p1, l1.p2, l2.p2);
    int o3 = orientation(l2.p1, l2.p2, l1.p1);
    int o4 = orientation(l2.p1, l2.p2, l1.p2);

    if (o1 != o2 && o3 != o4) {
        return true;
    }

    if (o1 == 0 && on_segment(l1.p1, l2.p1, l1.p2)) return true;
    if (o2 == 0 && on_segment(l1.p1, l2.p2, l1.p2)) return true;
    if (o3 == 0 && on_segment(l2.p1, l1.p1, l2.p2)) return true;
    if (o4 == 0 && on_segment(l2.p1, l1.p2, l2.p2)) return true;

    return false;
}

void doCollision(uint in1Index, uint fitnessIndex) {
    carData[in1Index] = startPoint.x;
    carData[in1Index + 1] = startPoint.y;
    carData[in1Index + 2] = startAngle;
    carData[in1Index + 3] = 0.0;
    carData[in1Index + 4] = 0.0;

    // reduce num laps by 2
    carFitness[fitnessIndex + 4] = carFitness[fitnessIndex + 4] - 2;
}

void calculateFitness(uint fitnessIndex) {
    carFitness[fitnessIndex + 5] = carFitness[fitnessIndex + 2] + carFitness[fitnessIndex + 4] * 50.0;
}

void main()
{
    uint index = gl_GlobalInvocationID.x;
    uint in1Index = index * numCarFloats;
    uint in2Index = index * numInputs;
    uint fitnessIndex = index * numFitnessFloats;

    float x = carData[in1Index];
    float y = carData[in1Index + 1];
    float angle = carData[in1Index + 2];
    float speed = carData[in1Index + 3];
    float acceleration = carData[in1Index + 4];

    if (calNewDistance == 1) {
        float dx = x - carFitness[fitnessIndex];
        float dy = y - carFitness[fitnessIndex + 1];
        carFitness[fitnessIndex + 2] += sqrt(dx * dx + dy * dy);
        carFitness[fitnessIndex] = x;
        carFitness[fitnessIndex + 1] = y;
    }

    float appliedForce = 0.0;
    float appliedTurning = 0.0;

    appliedForce += carInputs[in2Index] * engineForce;
    appliedForce -= carInputs[in2Index + 1] * engineForce;
    appliedForce += carInputs[in2Index + 4] * brakeForce * (speed > 0.0 ? -1.0 : 1.0);
    appliedTurning += carInputs[in2Index + 2] * maxTurnRate;
    appliedTurning -= carInputs[in2Index + 3] * maxTurnRate;

    float vvmaxS = (speed / maxSpeed) * (speed / maxSpeed);

    float airResistance = vvmaxS * engineForce;
    float totalForce = appliedForce - airResistance;

    float totalTurning = appliedTurning * (-2.5 * ((speed / maxSpeed) - 0.45) * ((speed / maxSpeed) - 0.45) + 1.0);

    
    acceleration = totalForce / carMass;
    angle += totalTurning * deltaTime;
    speed += acceleration * deltaTime;

    float oldX = x;
    float oldY = y;
    vec2 change = vec2(speed * cos(angle) * deltaTime, speed * sin(angle) * deltaTime);
    x += change.x;
    y += change.y;

    carData[in1Index] = x;
    carData[in1Index + 1] = y;
    carData[in1Index + 2] = angle;
    carData[in1Index + 3] = speed;
    carData[in1Index + 4] = acceleration;

    Line startLine = Line(insideStart, outsideStart);
    Line carLine = Line(vec2(oldX, oldY), vec2(x, y));
    if (intersect(startLine, carLine)) {
        float newL = length(change + startNormal);
        float maxL = max(length(change), length(startNormal));
        if (newL > maxL) {
            carFitness[fitnessIndex + 4] = carFitness[fitnessIndex + 4] + 1;
        } else if (newL < maxL) {
            doCollision(in1Index, fitnessIndex);
        }
    }

    for (uint i = 0; i < numInsideTrackPoints; i++) {
        uint inIndex = i * 2;
        Line trackLine = Line(vec2(insideTrack[inIndex], insideTrack[inIndex + 1]), vec2(insideTrack[(inIndex + 2) % (numInsideTrackPoints * 2)], insideTrack[(inIndex + 3) % (numInsideTrackPoints * 2)]));
        if (intersect(carLine, trackLine)) {
            doCollision(in1Index, fitnessIndex);
        }
    }
    for (uint i = 0; i < numOutsideTrackPoints; i++) {
        uint inIndex = i * 2;
        Line trackLine = Line(vec2(outsideTrack[inIndex], outsideTrack[inIndex + 1]), vec2(outsideTrack[(inIndex + 2) % (numOutsideTrackPoints * 2)], outsideTrack[(inIndex + 3) % (numOutsideTrackPoints * 2)]));
        if (intersect(carLine, trackLine)) {
            doCollision(in1Index, fitnessIndex);
        }
    }

    calculateFitness(fitnessIndex);
}