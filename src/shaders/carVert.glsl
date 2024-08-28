#version 430

layout(location = 0) in vec3 position;

uniform vec4 colourIn;

void main()
{
    gl_Position = vec4(position.x, position.y, 0.0, 1.0);
}