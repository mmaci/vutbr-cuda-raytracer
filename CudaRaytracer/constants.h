#ifndef CONSTANTS_H
#define CONSTANTS_H

// custom types
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef char int8;
typedef short int16;
typedef int int32;

const char APP_NAME[] = "Ray Tracer";

const uint32 WINDOW_WIDTH = 800;
const uint32 WINDOW_HEIGHT = 600;
const float WINDOW_ASPECT = WINDOW_WIDTH / WINDOW_HEIGHT;
const uint32 WINDOW_SIZE = WINDOW_WIDTH * WINDOW_HEIGHT;
const uint32 NUM_THREADS = 256;

#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define NO_HIT -1

enum Materials {
	MATERIAL_BLUE = 0,
	MATERIAL_RED,
	MATERIAL_GREEN,
	MATERIAL_YELLOW,

	NUM_MATERIALS
};

#define NUM_PLANES 4
#define NUM_SPHERES 2
#define NUM_LIGHTS 1

#endif
