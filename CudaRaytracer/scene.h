#ifndef SCENE_H
#define SCENE_H

#include "ray.h"
#include "sphere.h"
#include "plane.h"
#include "color.h"

namespace CUDA {

		struct Scene
		{
			Sphere* s;
			int countS;
			Plane* p;
			int countP;
			Color interSColor;
			float t;
			float3 point;
			float3 normal;
			//enum ShapeKind { skPlane, skSphere, skNone } kind;

		    __device__ Scene(){
				countS = 0;
				countP = 0;
			}
			__device__ bool add(Sphere &aSphere) {
				if (countS == 0) {
					countS++;
					//cudaMalloc((Sphere *)s,sizeof(Sphere));
				} else
				{
					countS++;
					//s = (Sphere*) cudaRealloc(s,sizeof(Sphere)*countS);
				}
				if (s == NULL) {
				  return false;				
				} else
				{
				   s[countS-1] = aSphere;
				   return true;
				}
				

			
			}

			__device__ bool add(Plane &aPlane){
				if (countP == 0) {
					countP++;
					//p = (Plane*) malloc(sizeof(Plane));
				} else
				{
					countP++;
					//p = (Plane*) realloc(p,sizeof(Plane)*countP);
				}
				if (p == NULL) {
				  return false;				
				} else
				{
				   p[countP-1] = aPlane;
				   return true;
				}
			}
			__device__ bool intersect(Ray const& ray) {
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
			    
			}
			
			__device__ ~Scene(){
				//cudaFree(p);
				//cudaFree(s);

			}

		};	

}

#endif