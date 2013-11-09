
#include "cuda.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <iostream>
#include "constants.h"

#include <GL/glew.h>
#include <glm.hpp>
#include <glut.h>
#include <cuda_gl_interop.h>

extern "C" void launchRTKernel(uchar4* , uint32, uint32);

/** @var GLuint pixel buffer object */
GLuint PBO;
	
/** @var GLuint texture buffer */
GLuint textureId;

/** @var cudaGraphicsResource_t cuda data resource */
cudaGraphicsResource_t cudaResourceBuffer;

/** @var cudaGraphicsResource_t cuda texture resource */
cudaGraphicsResource_t cudaResourceTexture;

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
   
	launchRTKernel(data, WINDOW_WIDTH, WINDOW_HEIGHT);

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
