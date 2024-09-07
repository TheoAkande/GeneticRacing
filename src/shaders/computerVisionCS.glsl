#version 430

layout(local_size_x = 1) in;

layout(binding = 0) buffer inputBuffer1 { float carData[]; };
layout(binding = 1) buffer inputBuffer2 { float insidePoints[]; };
layout(binding = 2) buffer inputBuffer3 { float outsidePoints[]; };
layout(binding = 3) buffer inputBuffer4 {float visionAngles[]; };
layout(binding = 4) buffer outputBuffer {float distances[]; };

uniform int numCarFloats;
uniform int numInsidePoints;
uniform int numOutsidePoints;
uniform int numComputerVisionAngles;

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

float dist(Line carLine, Line trackLine, float minDistance) {
    // find the intersection point (and distance to it)
    float x1 = carLine.p1.x;
    float y1 = carLine.p1.y;

    float x2 = carLine.p2.x;
    float y2 = carLine.p2.y;

    float x3 = trackLine.p1.x;
    float y3 = trackLine.p1.y;

    float x4 = trackLine.p2.x;
    float y4 = trackLine.p2.y;

    float dx1 = x2 - x1;
    if (dx1 == 0.0) {
        dx1 = 0.0001;
    }
    float dy1 = y2 - y1;

    float dx2 = x4 - x3;
    if (dx2 == 0.0) {
        dx2 = 0.0001;
    }
    float dy2 = y4 - y3;

    float m1 = dy1 / dx1;
    float c1 = y1 - m1 * x1;

    float m2 = dy2 / dx2;
    float c2 = y3 - m2 * x3;

    float x = (c2 - c1) / (m1 - m2);
    float y = m1 * x + c1;

    float distance = sqrt((x - carLine.p1.x) * (x - carLine.p1.x) + (y - carLine.p1.y) * (y - carLine.p1.y));
    if (distance < minDistance) {
        minDistance = distance;
    }
    return minDistance;
}

void main()
{
    uint carIndex = gl_GlobalInvocationID.x;
    uint angleIndex = gl_GlobalInvocationID.y;
    uint inIndex = carIndex * numCarFloats;
    uint outIndex = carIndex * numComputerVisionAngles + angleIndex;

    float x = carData[inIndex];
    float y = carData[inIndex + 1];
    float angle = carData[inIndex + 2] + visionAngles[angleIndex];

    vec2 carPos = vec2(x, y);
    vec2 projectedPoint = carPos + vec2(cos(angle), sin(angle)) * 5.0;
    Line carLine = Line(carPos, projectedPoint);

    // check if the line intersects with each track line
    float minDistance = 1000.0;

    for (int i = 0; i < numInsidePoints; i++) {
        vec2 p1 = vec2(insidePoints[i * 2], insidePoints[i * 2 + 1]);
        vec2 p2 = vec2(insidePoints[(i + 1) % numInsidePoints * 2], insidePoints[(i + 1) % numInsidePoints * 2 + 1]);
        Line trackLine = Line(p1, p2);

        if (intersect(carLine, trackLine)) {
            minDistance = dist(carLine, trackLine, minDistance);
        }
    }
    for (int i = 0; i < numOutsidePoints; i++) {
        vec2 p1 = vec2(outsidePoints[i * 2], outsidePoints[i * 2 + 1]);
        vec2 p2 = vec2(outsidePoints[(i + 1) % numOutsidePoints * 2], outsidePoints[(i + 1) % numOutsidePoints * 2 + 1]);
        Line trackLine = Line(p1, p2);

        if (intersect(carLine, trackLine)) {
            minDistance = dist(carLine, trackLine, minDistance);
        }
    }

    if (angleIndex == 0) {
        float speed = carData[inIndex + 3];
        float brakingDistance = speed * speed / (2.0 * 0.3);
        minDistance = brakingDistance;
    }

    distances[outIndex] = minDistance;
}

