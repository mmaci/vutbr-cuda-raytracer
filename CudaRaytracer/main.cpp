
#include "cuda.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <iostream>
#include "constants.h"

#include <GL/glew.h>
#include <glm.hpp>
#include <glut.h>
#include <cuda_gl_interop.h>

#include "ray.h"
#include "sphere.h"
#include "color.h"

#include <vector_types.h>
#include <vector_functions.h>
#include "mathematics.h"

#define NUM_SPHERES 2

Sphere spheres[NUM_SPHERES];
GLuint PBO;

// Camera parameters -----------------------------
float3 a; float3 b; float3 c; 
float3 campos; 
float cameraRotation = 0.f;
float cameraDistance = 75.f;
float cameraHeight = 25.f;

extern "C" void RayTraceImage(float3* dataOut, Sphere* spheres, uint32 numSpheres,
		               float3 a, float3 b, float3 c, 
		               float3 campos);


void rayTrace()
{
	float3* outData;
	cudaGLMapBufferObject((void**)&outData, PBO);

	RayTraceImage(outData, spheres, NUM_SPHERES, a, b, c, campos);

	cudaGLUnmapBufferObject(PBO);	
}


void display(void)
{	
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	rayTrace();
	
	glutSwapBuffers();
	glutPostRedisplay();
}

void init()
{
	// initialize the PBO for transferring data from CUDA to openGL
	uint32 num = WIDTH * HEIGHT;
	uint32 dataSize = sizeof(float3) * num;
	void *data = malloc(dataSize);

	// create buffer object
	glGenBuffers(1, &PBO);
	glBindBuffer(GL_ARRAY_BUFFER, PBO);
	glBufferData(GL_ARRAY_BUFFER, dataSize, data, GL_DYNAMIC_DRAW);
	free(data);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	cudaGLRegisterBufferObject(PBO);	
}

void initCamera()
{
	campos = make_float3(cos(cameraRotation)*cameraDistance, cameraHeight, -sin(cameraRotation)* cameraDistance);
	float3 cam_dir = -1.f * campos;
	math::normalize(cam_dir);
	float3 cam_up  = make_float3(0,1,0);
	float3 cam_right = math::crossProduct(cam_dir,cam_up);
	math::normalize(cam_right);

	cam_up = -1.f * math::crossProduct(cam_dir,cam_right);
	math::normalize(cam_up);
	
	float FOV = 60.0f;
	float theta = (FOV*3.1415*0.5) / 180.0f;
	float half_width = tanf(theta);
	float aspect = (float)WIDTH / (float)HEIGHT;

	float u0 = -half_width * aspect;
	float v0 = -half_width;
	float u1 =  half_width * aspect;
	float v1 =  half_width;
	float dist_to_image = 1;

	a = (u1-u0)*cam_right;
	b = (v1-v0)*cam_up;
	c = campos + u0*cam_right + v0*cam_up + dist_to_image*cam_dir;
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WIDTH, HEIGHT);	
	glutCreateWindow("Ray tracer");
	
	glewInit();
	init();
	initCamera();

	glutDisplayFunc(display);
	
	glutMainLoop();
	cudaThreadExit();	

	return EXIT_SUCCESS;
}
