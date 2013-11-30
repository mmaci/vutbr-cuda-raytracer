
#include "cuda.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <iostream>
#include "constants.h"

#include <GL/glew.h>
#include <glm.hpp>
#include <glut.h>
#include <cuda_gl_interop.h>

#include "mathematics.h"

#include "scene.h"
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "phong.h"

Sphere* devSpheres;
Plane* devPlanes;
Camera* devCamera;
SceneStats* devSceneStats;

/** @var GLuint pixel buffer object */
GLuint PBO;

/** @var GLuint texture buffer */
GLuint textureId;

/** @var cudaGraphicsResource_t cuda data resource */
cudaGraphicsResource_t cudaResourceBuffer;

/** @var cudaGraphicsResource_t cuda texture resource */
cudaGraphicsResource_t cudaResourceTexture;

extern "C" void launchRTKernel(uchar4* , uint32, uint32, Sphere*, Plane*,SceneStats*, Camera*);

/**
* 1. Maps the the PBO (Pixel Buffer Object) to a data pointer
* 2. Launches the kernel
* 3. Unmaps the PBO
*/ 
void runCuda()
{	
	uchar4* data;
	size_t numBytes;

	cudaGraphicsMapResources(1, &cudaResourceBuffer, 0);
	// cudaGraphicsMapResources(1, &cudaResourceTexture, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&data, &numBytes, cudaResourceBuffer);

	launchRTKernel(data, WINDOW_WIDTH, WINDOW_HEIGHT, devSpheres, devPlanes, devSceneStats, devCamera);

	cudaGraphicsUnmapResources(1, &cudaResourceBuffer, 0);
	// cudaGraphicsUnmapResources(1, &cudaResourceTexture, 0);	
}

/**
* Display callback
* Launches both the kernel and draws the scene
*/
void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	// run the Kernel
	runCuda();

	// and draw everything
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f,0.0f,0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f,1.0f,0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f,1.0f,0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f,0.0f,0.0f);
	glEnd();

	glutSwapBuffers();
	glutPostRedisplay();  
}

/**
* Initializes the CUDA part of the app
*
* @param int number of args
* @param char** arg values
*/
void initCuda(int argc, char** argv)
{	  	
	int sizeData = sizeof(uchar4) * WINDOW_SIZE;

	// Generate, bind and register the Pixel Buffer Object (PBO)
	glGenBuffers(1, &PBO);    
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, PBO);    
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sizeData, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);    

	cudaGraphicsGLRegisterBuffer(&cudaResourceBuffer, PBO, cudaGraphicsMapFlagsNone);

	// Generate, bind and register texture
	glEnable(GL_TEXTURE_2D);   
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0); // unbind

	// cudaGraphicsGLRegisterImage(&cudaResourceTexture, textureId, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);

	runCuda();
}

void initScene(Scene* scene) {

	PhongInfo matBlue(Color(0.f, 0.f, 1.f),Color(1.f, 1.f, 1.f),Color(0.15, 0.1, 0.1),0);
	PhongInfo matRed(Color(1.f, 0.f, 0.f), Color(1.f, 1.f, 1.f), Color(0.1, 0.05, 0.05), 15);
	PhongInfo matGreen(Color(0.f, 1.f, 0.f), Color(1.f, 1.f, 1.f), Color(0.25, 0, 0), 5);


	Sphere s(make_float3(8.f, 4.f, 0.f), 2.f,matRed);
	scene->add(s);
	Sphere s1(make_float3(4.f, -5.f, 4.f), 4.f,matGreen);
	scene->add(s1);

	Plane p(make_float3(1.f, 5.f, 10.f), make_float3(5.f, 0.f, 0.f),matBlue);
	scene->add(p);

	scene->getCamera()->lookAt(make_float3(2.f, 3.f, -7.f),  // eye
		make_float3(5.f, 0.f, 1.f),   // target
		make_float3(0.f, 1.f, 0.f),   // sky
		30, (float)WINDOW_WIDTH/WINDOW_HEIGHT);

	cudaMalloc((void***) &devSpheres, scene->getSphereCount() * sizeof(Sphere));
	cudaMalloc((void***) &devPlanes, scene->getPlaneCount() * sizeof(Plane));
	cudaMalloc((void***) &devCamera, sizeof(Camera));
	cudaMalloc((void***) &devSceneStats, sizeof(SceneStats));

	cudaMemcpy(devPlanes, scene->getPlanes(), scene->getPlaneCount() * sizeof(Plane), cudaMemcpyHostToDevice);
	cudaMemcpy(devSpheres, scene->getSpheres(), scene->getSphereCount() * sizeof(Sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(devCamera, scene->getCamera(), sizeof(Camera), cudaMemcpyHostToDevice);
	cudaMemcpy(devSceneStats, scene->getSceneStats() , sizeof(SceneStats), cudaMemcpyHostToDevice);
}

/**
* Initializes the OpenGL part of the app
*
* @param int number of args
* @param char** arg values
*/
void initGL(int argc, char** argv)
{	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow(APP_NAME);
	glutDisplayFunc(display);

	// check for necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0")) {
		std::cerr << "ERROR: Support for necessary OpenGL extensions missing.";
		return;
	}

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);   
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// set matrices
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);   	
}

/**
* Main
*
* @param int number of args
* @param char** arg values
*/
int main(int argc, char** argv)
{	 
	initGL(argc, argv);
	Scene s;
	initScene(&s);

	initCuda(argc, argv);  



	glutDisplayFunc(display);
	glutMainLoop();

	cudaThreadExit();  

	return EXIT_SUCCESS;
}
