#version 430

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 colour;

out vec4 varyingColour;

void main()
{
    gl_Position = vec4(position.x, position.y, 0.0, 1.0);
    varyingColour = colour;
}