#include <vector>
#include <vector_types.h>
#include "constants.h"
#include <algorithm>

const uint32 SPLIT_LIMIT = 3;

struct Obj {

	float x;
	float y;
	float z;
	float radius;

	Sphere sphere;
};

struct BVHnode {

	float3 max;
	float3 min;

	BVHnode* prev;
	BVHnode* next;
	BVHnode* parent;

	std::vector<Obj> leaves;

	char nextAxis(char axis) {
		return 'x';

		switch (axis) {
			case 'x': return 'y';
			case 'y': return 'x';
			default: return 'x';
		}		
	}	

	void buildBVH(std::vector<Obj> objects, BVHnode* p, uint32 start, uint32 end, char axis)
	{		
		uint32 count = end - start;
		std::vector<Obj> newList;
		parent = p;

		// error case
		if (end < start)
		{
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

		// finish state
		if (count < SPLIT_LIMIT)
		{
			min.x = objects[start].x - objects[start].radius;
			max.x = objects[start].x + objects[start].radius;
			min.y = objects[start].y - objects[start].radius;
			max.y = objects[start].y + objects[start].radius;
			min.z = objects[start].z - objects[start].radius;
			max.z = objects[start].z + objects[start].radius;

			prev = nullptr;
			next = nullptr;
	
			uint32 n = 0;
			for (uint32 loop = start; loop <= end; loop++)
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

				leaves.push_back(objects[loop]);
				n++;
			}
			return;
		}

		// normal case

		// copy all the objects in a new list
		for (uint32 loop = start; loop <= end; loop++) {
			newList.push_back(objects[loop]);
		}

		// sort them by whatever axis we want
		switch(axis) {
			case 'x':
				std::sort(newList.begin(), newList.end(), [](Obj const& p1, Obj const& p2){ return p1.x > p2.x; });
				break;
			case 'y':
				std::sort(newList.begin(), newList.end(), [](Obj const& p1, Obj const& p2){ return p1.y > p2.y; });
				break;
			case 'z':
				std::sort(newList.begin(), newList.end(), [](Obj const& p1, Obj const& p2){ return p1.z > p2.z; });
				break;
		}

		// divide them into 2 parts
		uint32 center = count / 2;

		min.x = newList[0].x - newList[0].radius;
		max.x = newList[0].x + newList[0].radius;
		min.y = newList[0].y - newList[0].radius;
		max.y = newList[0].y + newList[0].radius;
		min.z = newList[0].z - newList[0].radius;
		max.z = newList[0].z + newList[0].radius;    

		prev = new BVHnode;
		prev->buildBVH(newList, this, 0, center, nextAxis(axis));

		next = new BVHnode;
		next->buildBVH(newList, this, center + 1, count, nextAxis(axis)); 

		if (parent) {
			if (min.x < parent->min.x)
				parent->min.x = min.x;

			if (max.x > parent->max.x)
				parent->max.x = max.x;

			if (min.y < parent->min.y)
				parent->min.y = min.y;

			if (max.y > parent->max.y)
				parent->max.y = max.y;

			if (min.z < parent->min.z)
				parent->min.z = min.z;

			if (max.z > parent->max.z)
				parent->max.z = max.z;
		}
	}
};
