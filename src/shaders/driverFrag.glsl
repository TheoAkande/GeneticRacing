#version 430

out vec4 colour;

in vec4 varyingColour;

void main() 
{
    colour = varyingColour;
}