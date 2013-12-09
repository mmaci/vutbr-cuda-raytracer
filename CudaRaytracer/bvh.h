#include <vector>
#include <vector_types.h>
#include "constants.h"

const uint32 SPLIT_LIMIT = 3;

struct Primitive {

	float x;
	float y;
	float z;
	float radius;

};

class BVHnode {

private:

	float3 max;
	float3 min;

	BVHnode* prev;
	BVHnode* next;
	BVHnode* parent;

	std::vector<Primitive> primitives;	

public:

	char nextAxis(char axis) {
		switch (axis) {
			case 'x': return 'y';
			case 'y': return 'z';
			case 'z': default: return 'x';
		}		
	}

	void buildBVH(std::vector<Primitive> objects, BVHnode* parent, uint32 start, uint32 end, char axis)
	{		
		uint32 count = end - start;
		
		Primitive newPrimitive;
		std::vector<Primitive> newList;

		if (objects.empty() || end < start) {
			max.x = 0.f;
			max.y = 0.f;
			max.z = 0.f;

			min.x = 0.f;
			min.y = 0.f;
			min.z = 0.f;

			prev = nullptr;
			next = nullptr;			

			return;
		}

		if (count < SPLIT_LIMIT) {
			min.x = objects[start].x - objects[start].radius;
			max.x = objects[start].x + objects[start].radius;
			min.y = objects[start].y - objects[start].radius;
			max.y = objects[start].y + objects[start].radius;
			min.z = objects[start].z - objects[start].radius;
			max.z = objects[start].z + objects[start].radius;

			prev = nullptr;
			next = nullptr;
	

		for (int loop = start; loop <= end; loop++)
		{
			// X
			 
			if (objects[loop].x - objects[loop].radius < min.x)
				min.x = objects[loop].x - objects[loop].radius;
			if (objects[loop].x + objects[loop].radius > max.x)
				max.x = objects[loop].x + objects[loop].radius;
			if (parent && (min.x < parent->min.x))
				parent->min.x = min.x;			
			if (parent && max.x > parent->max.x)
				parent->max.x = max.x;

			// Y

			if (objects[loop].y - objects[loop].radius < min.y)
				min.y = objects[loop].y - objects[loop].radius;
			if (objects[loop].y + objects[loop].radius > max.y)
				max.y = objects[loop].y + objects[loop].radius;
			if (parent && min.y < parent->min.y)
				parent->min.y = min.y;			
			if (parent && max.y > parent->max.y)
				parent->max.y = max.y;

			// Z

			if (objects[loop].z - objects[loop].radius < min.z)
				min.z = objects[loop].z - objects[loop].radius;
			if (objects[loop].z + objects[loop].radius > max.z)
				max.z = objects[loop].z + objects[loop].radius;
			if (parent && min.z < parent->min.z)
				parent->min.z = min.z;			
			if (parent && max.z > parent->max.z)
				parent->max.z = max.z;

			primitives.push_back(objects[loop]);			
		}
		return;
		}

		for (int loop = start; loop <= end; loop++) {
			newList.push_back(objects[loop]);
		}

		switch(axis) {
			case 'x':
				// sort along x axis
				break;
			case 'y':
				// sort along y axis
				break;
			case 'z':
				// sort along z axis
				break;
		}

		int center = (int) (count * 0.5f);

		min.x = newList[0].x - newList[0].radius;
		max.x = newList[0].x + newList[0].radius;
		min.y = newList[0].y - newList[0].radius;
		max.y = newList[0].y + newList[0].radius;
		min.z = newList[0].z - newList[0].radius;
		max.z = newList[0].z + newList[0].radius;
    

	buildBVH(newList, this, 0, center, nextAxis(axis));
    buildBVH(newList, this, center + 1, count, nextAxis(axis)); 

    if (parent && min.x < parent->min.x)
        parent->min.x = min.x;

    if (parent && max.x > parent->max.x)
        parent->max.x = max.x;

    if (parent && min.y < parent->min.y)
        parent->min.y = min.y;

    if (parent && max.y > parent->max.y)
        parent->max.y = max.y;

    if (parent && min.z < parent->min.z)
        parent->min.z = min.z;

    if (parent && max.z > parent->max.z)
        parent->max.z = max.z;
	}
};
