#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "sphere.h"
#include "plane.h"
#include "camera.h"

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

		/*bool intersect(Ray const& ray) {
			int i;
			float st,pt;
			st = 0.f;
			pt = 0.f;
			int maxPi,maxSi;
			float tmp;
			for (i=0;i<countS;i++){
				tmp = s[i].intersect(ray);
				if (st < tmp){
					st = tmp;
					maxSi =i;
				}				
			}
			for (i=0;i<countP;i++){
				tmp = p[i].intersect(ray);
				if (pt < tmp){
					pt = tmp;
					maxPi = i;
				}				
			}
			if ((pt==0.f) && (st == 0.f)){ //miss
				t = 0.f;
				//kind = skNone;
				return false;
			} else if (pt > st) //plane hit
			{
				point = ray.getPoint(pt);
				normal = p[maxPi].normal;
				t = pt;
				interSColor = p[maxPi].color;

	
			}else if (st >= pt) //sphere hit
			{
				point = ray.getPoint(st);
				normal = s[maxSi].getNormal(point);
				t = st;
				interSColor = s[maxSi].color;

			};

			return true;
			    
		}*/
					
	private:
		std::vector<Sphere> spheres;
		std::vector<Plane> planes;
		Camera camera;

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