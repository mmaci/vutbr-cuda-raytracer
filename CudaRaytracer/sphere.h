#ifndef SPHERE_H
#define SPHERE_H

#include <vector>
#include "mathematics.h"
#include <GL/glew.h>


class Sphere
{
	public:
		__device__ Sphere(float radius, unsigned int rings, unsigned int sectors)
		{
			
		}

		void draw(float x, float y, float z)
		{
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glTranslatef(x,y,z);

			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_NORMAL_ARRAY);
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);

			glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);
			glNormalPointer(GL_FLOAT, 0, &normals[0]);
			glTexCoordPointer(2, GL_FLOAT, 0, &texcoords[0]);
			glDrawElements(GL_QUADS, indices.size(), GL_UNSIGNED_SHORT, &indices[0]);
			glPopMatrix();
		}

	private:
		std::vector<float> vertices;
		std::vector<float> normals;
		std::vector<float> texcoords;
		std::vector<uint16> indices;
};

#endif
