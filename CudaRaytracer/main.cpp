
#include "cuda.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <iostream>
#include "constants.h"

#include <GL/glew.h>
#include <glm.hpp>
#include <glut.h>
#include <cuda_gl_interop.h>

#include <vector_types.h>
#include <vector_functions.h>
#include <vector>

#include "ray.h"
#include "sphere.h"
#include "color.h"
#include "mathematics.h"
#include <time.h>

extern "C" void launchRTKernel(uchar4* , uint32, uint32);

/** @var GLuint pixel buffer object */
GLuint PBO;
	
/** @var GLuint texture buffer */
GLuint textureId;

/**
 * 1. Maps the the PBO (Pixel Buffer Object) to a data pointer
 * 2. Launches the kernel
 * 3. Unmaps the PBO
 */ 
void runCuda()
{
	uchar4* data = nullptr; 
	cudaGLMapBufferObject((void**)&data, PBO);
   
	launchRTKernel(data, WINDOW_WIDTH, WINDOW_HEIGHT);
  
	cudaGLUnmapBufferObject(PBO);
}


/**
 * Display callback
 * Launches both the kernel and draws the scene
 */
void display()
{
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
	int numValues = WINDOW_SIZE * sizeof(uchar4);
    int sizeData = sizeof(GLubyte) * numValues;
     
    // Generate, bind and register the PBO
    glGenBuffers(1, &PBO);    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);    
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeData, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(PBO);
	
	glEnable(GL_TEXTURE_2D);   
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  
	runCuda();
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
	initCuda(argc, argv);  
   
	glutDisplayFunc(display);
	glutMainLoop();
     
	cudaThreadExit();  

	return EXIT_SUCCESS;
}
