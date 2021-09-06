#pragma once

#include "cl_types.h"

typedef struct ClSphConfig {
  ALIGNED_(4) float dt;
  ALIGNED_(4) float scale;
  ALIGNED_(8) size_t iteration;
  ALIGNED_(16) __device_float3 constForce;
  ALIGNED_(16) __device_float3 minBound;
  ALIGNED_(16) __device_float3 maxBound;
} ClSphConfig;

typedef enum ALIGNED_(4) ClSphType { Fluid, Obstacle } ClSphType;

typedef struct ClSphParticle {
  ALIGNED_(8) size_t id;
  ClSphType type;
  ALIGNED_(4) float mass;
  ALIGNED_(16) __device_float3 position;
  ALIGNED_(16) __device_float3 velocity;
} ClSphParticle;

typedef struct ClSphAtom {
  ClSphParticle particle;
  ALIGNED_(16) __device_float3 pStar;
  ALIGNED_(16) __device_float3 deltaP;
  //	ALIGNED_(16) __device_float3 omega;
  ALIGNED_(4) float lambda;
  ALIGNED_(8) size_t zIndex;
} ClSphAtom;

typedef struct ClMcConfig {
  ALIGNED_(4) float sampleResolution;
  ALIGNED_(4) float particleSize;
  ALIGNED_(4) float particleInfluence;
  ALIGNED_(4) float isolevel;
} ClMcConfig;

typedef struct ClSphTraiangle {
  ALIGNED_(16) __device_float3 a;
  ALIGNED_(16) __device_float3 b;
  ALIGNED_(16) __device_float3 c;
} ClSphTraiangle;

typedef struct ClSphResponse {
  ALIGNED_(16) __device_float3 position;
  ALIGNED_(16) __device_float3 velocity;
} ClSphResponse;

const __device_constant size_t _SIZES[] = {
    sizeof(size_t),          sizeof(__device_uint), sizeof(__device_uint3), sizeof(float),
    sizeof(__device_float3), sizeof(ClSphType),     sizeof(ClSphConfig),    sizeof(ClMcConfig),
    sizeof(ClSphAtom),       sizeof(ClSphParticle), sizeof(ClSphResponse),
};

const __device_constant size_t _SIZES_LENGTH = sizeof(_SIZES) / sizeof(_SIZES[0]);
