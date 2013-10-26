
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "contants.h"

#include <GL/glew.h>
#include <glm.hpp>
#include <glut.h>


void display(void)
{

}

int main(int argc, char** argv)
{
	glutInit (&argc, argv);
	// specify the display mode to be RGB and single buffering:
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window position:
	glutInitWindowPosition(100,100);
	// specify the initial window size:
	glutInitWindowSize(800, 600);
	// create the window and set title:
	glutCreateWindow("Ray tracer");
	
	glutDisplayFunc(display);
	
	glutMainLoop();
	
	return EXIT_SUCCESS;
}
