#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "sphere.h"
#include "plane.h"
#include "camera.h"

struct SceneStats{
	int SphereCount;
	int PlaneCount;
};

class Scene
{
public: 
	uint32 getSphereCount() const { return spheres.size(); }
	uint32 getPlaneCount() const { return planes.size(); }

	void add(Sphere s) { spheres.push_back(s); }
	void add(Plane p){ planes.push_back(p); }

	Camera* getCamera() { return &camera; }
	Sphere* getSpheres() { return &spheres[0]; }
	Plane* getPlanes() { return &planes[0];	}
	SceneStats* getSceneStats(){
		sceneStats.PlaneCount = planes.size();
		sceneStats.SphereCount = spheres.size();
		return &sceneStats;
	};

private:
	std::vector<Sphere> spheres;
	std::vector<Plane> planes;
	Camera camera;
	SceneStats sceneStats;
	/*
	Sphere* s;
	int countS;
	Plane* p;
	int countP;
	Color interSColor;
	float t;
	float3 point;
	float3 normal;*/
	//enum ShapeKind { skPlane, skSphere, skNone } kind;

};

#endif