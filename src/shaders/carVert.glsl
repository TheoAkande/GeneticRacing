#version 430

layout(location = 0) in vec3 position;

void main()
{
    gl_Position = vMatrix * vec4(position.x, position.y, 0.0, 1.0);
}