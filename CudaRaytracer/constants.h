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

#endif
