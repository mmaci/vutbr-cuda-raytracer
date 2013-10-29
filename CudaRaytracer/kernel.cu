
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "constants.h"

#include <GL/glew.h>
#include <glm.hpp>
#include <glut.h>

#include "ray.h"
#include "pixel.h"


__device__ 
ray generateRay(std::pair<float, float> viewport, std::pair<float, float> pixel)
{			
	float x = 2.0f * pixel.first / viewport.first - 1.0f;
	float y = 2.0f * pixel.second / viewport.second - 1.0f;		
				
	vector3 origin(x, y, -1.0f);
	vector3 direction(x, y, 1.0f);

	return ray(origin, vector3(direction - origin).normalize());
}

__global__ 
void rayTraceKernel(float3* pos)
{
	uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 j = blockIdx.y * blockDim.y + threadIdx.y;
			
	ray r = generateRay(std::pair<float, float>(WIDTH, HEIGHT), std::pair<float, float>(i, j));

	// check all spheres if they intersect with the generated ray

	// if they do, compute the color of the pixel			
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
}

int main(int argc, char** argv)
{
	glutInit (&argc, argv);
	// specify the display mode to be RGB and single buffering:
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window position:
	glutInitWindowPosition(100,100);
	// specify the initial window size:
	glutInitWindowSize(WIDTH, HEIGHT);
	// create the window and set title:
	glutCreateWindow("Ray tracer");
	
	glutDisplayFunc(display);
	
	glutMainLoop();
	
	return EXIT_SUCCESS;
}
