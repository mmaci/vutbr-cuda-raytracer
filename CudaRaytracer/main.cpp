﻿
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

#include <chrono>


#include "bvh.h"


SceneStats* dev_sceneStats;

Scene scene;

/** @var GLuint pixel buffer object */
GLuint PBO;

/** @var GLuint texture buffer */
GLuint textureId;

/** @var cudaGraphicsResource_t cuda data resource */
cudaGraphicsResource_t cudaResourceBuffer;

/** @var cudaGraphicsResource_t cuda texture resource */
cudaGraphicsResource_t cudaResourceTexture;

extern "C" void launchRTKernel(uchar3* , uint32, uint32, Sphere*, Plane*, PointLight*, PhongMaterial*, Camera*);

float deltaTime = 0.0f;
float fps = 0.0f;
float delta;

/**
* 1. Maps the the PBO (Pixel Buffer Object) to a data pointer
* 2. Launches the kernel
* 3. Unmaps the PBO
*/ 
void runCuda()
{		
	uchar3* data;
	size_t numBytes;

	cudaGraphicsMapResources(1, &cudaResourceBuffer, 0);
	// cudaGraphicsMapResources(1, &cudaResourceTexture, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&data, &numBytes, cudaResourceBuffer);

	launchRTKernel(data, WINDOW_WIDTH, WINDOW_HEIGHT, scene.getSpheres(), scene.getPlanes(), scene.getLights(), scene.getMaterials(), scene.getCamera());		

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

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	// run the Kernel
	runCuda();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	// and draw everything
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f,0.0f,0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f,1.0f,0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f,1.0f,0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f,0.0f,0.0f);
	glEnd();

	glutSwapBuffers();
	glutPostRedisplay();  

	

	float delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;

	deltaTime += delta;
	deltaTime /= 2.0f;
	fps = 1.0f / deltaTime;

	std::cout << std::fixed << fps << std::endl;
}

/**
* Initializes the CUDA part of the app
*
* @param int number of args
* @param char** arg values
*/
void initCuda(int argc, char** argv)
{	  	
	int sizeData = sizeof(uchar3) * WINDOW_SIZE;

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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0); // unbind

	// cudaGraphicsGLRegisterImage(&cudaResourceTexture, textureId, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	cudaError_t stat;
	size_t myStackSize = STACK_SIZE;
	stat = cudaDeviceSetLimit(cudaLimitStackSize, myStackSize);
	runCuda();
}

void initMaterials() {
	Color blue; blue.set(0.f, 0.f, 1.f);
	Color red; red.set(1.f, 0.f, 0.f);
	Color green; green.set(0.f, 1.f, 0.f);
	Color yellow; yellow.set(1.f, 1.f, 0.f);
	Color white; white.set(1.f, 1.f, 1.f);

	Color c1; c1.set(0.15, 0.1, 0.1);
	Color c2; c2.set(0.1, 0.05, 0.05);
	Color c3; c3.set(0.25, 0, 0);
	Color c4; c4.set(0.15, 0.1, 0.1);

	PhongMaterial matBlue; matBlue.set(blue, white, c1, 15);
	scene.add(matBlue); // MATERIAL_BLUE
	PhongMaterial matRed; matRed.set(red, white, c2, 15);
	scene.add(matRed); // MATERIAL_RED
	PhongMaterial matGreen; matGreen.set(green, white, c3, 20, 0.5);
	scene.add(matGreen); // MATERIAL_GREEN
	PhongMaterial matYellow; matYellow.set(yellow, white, c4, 15);
	scene.add(matYellow); // MATERIAL_YELLOW
}

void initSpheres() {
	Sphere s1;
	s1.set(make_float3(-1.7, 4, 0), 1.6, MATERIAL_BLUE);
	scene.add(s1);

	Sphere s2;
	s2.set(make_float3(1.7, 4, 0), 1.6, MATERIAL_RED);
	scene.add(s2);

	std::vector<Sphere> spheres = scene.getSphereVector();
	std::vector<Obj> objects;
	for (std::vector<Sphere>::iterator it = spheres.begin(); it != spheres.end(); ++it) {
		Sphere sphere = *it;
		Obj o;
		o.sphere = sphere;
		o.x = sphere.center.x;
		o.y = sphere.center.y;
		o.z = sphere.center.z;
		o.radius = sphere.radius;

		objects.push_back(o);
	}

	BVHnode tree;
	tree.buildBVH(objects, nullptr, 0, objects.size() - 1, 'x');


}

void initPlanes() {
	Plane p1; p1.set(make_float3(0, 0, 1), make_float3(0, 0, 15), MATERIAL_BLUE);
	scene.add(p1); // vzadu
	Plane p2; p2.set(make_float3(0, 1, 0), make_float3(0, -1.5, 0), MATERIAL_RED);
	scene.add(p2); // podlaha
	Plane p3; p3.set(make_float3(1, 0, 0), make_float3(-10, 0, 0), MATERIAL_GREEN);
	scene.add(p3); // leva strana
	Plane p4; p4.set(make_float3(-1, 0, 0), make_float3(10, 0, 0), MATERIAL_YELLOW);
	scene.add(p4); // prava strana
}

void initLights() {
	Color white; white.set(1.f, 1.f, 1.f);
	PointLight l1; l1.set(make_float3(-2.f, 10.f, -15.f), white);
	scene.add(l1);
}

void initScene() {

	initMaterials();
	initSpheres();
	initPlanes();
	initLights();
	
	//scene.add(PointLight(make_float3(0, 10, 0), Color(1, 1, 1)));

	/*Sphere s(make_float3(8.f, -4.f, 0.f), 2.f, matRed);
	scene.add(s);
	Sphere s1(make_float3(4.f, 0.f, 4.f), 4.f, matGreen);
	scene.add(s1);	
	Plane p(make_float3(7.f, 10.f, -10.f), make_float3(5.f, 0.f, 0.f), matBlue);
	scene.add(p);
	PointLight l(make_float3(1.f, 5.f, 4.f), Color(1.f, 1.f, 1.f));
	scene.add(l);
	PointLight l2(make_float3(9.f, 10.f, 1.f), Color(1.f, 1.f, 1.f));
	scene.add(l2);*/
	
	scene.getCamera()->init();
	scene.getCamera()->lookAt(make_float3(3.f, 3.f, -7.f),  // eye
		make_float3(0.f, 0.f, 1.f),   // target
		make_float3(0.f, 1.f, 0.f),   // sky
		30, (float)WINDOW_WIDTH/WINDOW_HEIGHT);

	// cudaMalloc((void***) &devSpheres, scene.getSphereCount() * sizeof(Sphere));
	// cudaMalloc((void***) &devPlanes, scene.getPlaneCount() * sizeof(Plane));
	// cudaMalloc((void***) &devLights, scene.getLightCount() * sizeof(PointLight));
	// cudaMalloc((void***) &devCamera, sizeof(Camera));
	// cudaMalloc((void***) &dev_sceneStats, sizeof(SceneStats));

	// cudaMemcpy(devPlanes, scene.getPlanes(), scene.getPlaneCount() * sizeof(Plane), cudaMemcpyHostToDevice);
	// cudaMemcpy(devSpheres, scene.getSpheres(), scene.getSphereCount() * sizeof(Sphere), cudaMemcpyHostToDevice);
	// cudaMemcpy(devLights, scene.getLights(), scene.getLightCount() * sizeof(PointLight), cudaMemcpyHostToDevice);
	// cudaMemcpy(devCamera, scene.getCamera(), sizeof(Camera), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_sceneStats, scene.getSceneStats() , sizeof(SceneStats), cudaMemcpyHostToDevice);	
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
	initScene();

	initCuda(argc, argv);  



	glutDisplayFunc(display);
	glutMainLoop();

	cudaThreadExit();  

	return EXIT_SUCCESS;
}
