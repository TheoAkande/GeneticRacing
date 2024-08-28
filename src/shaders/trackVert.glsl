#version 430

layout(location = 0) in vec2 position;

uniform mat4 vMatrix;

void main()
{
    gl_Position = vMatrix * vec4(position, 0.0, 1.0);
}