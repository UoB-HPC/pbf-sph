
#include "curves.h"
#include "mc_constants.h"
#include "oclsph_type.h"
#include "sph_constants.h"

#ifndef SPH_H
#error SPH_H is not set
#endif

const constant float CorrDeltaQH = CorrDeltaQ * SPH_H;
const constant float H2 = SPH_H * 2;
const constant float HH = SPH_H * SPH_H;
const constant float HHH = SPH_H * SPH_H * SPH_H;

const constant uint3 NEIGHBOUR_OFFSETS[27] = {
    (uint3)(-1, -1, -1), (uint3)(+0, -1, -1), (uint3)(+1, -1, -1), (uint3)(-1, +0, -1), (uint3)(+0, +0, -1),
    (uint3)(+1, +0, -1), (uint3)(-1, +1, -1), (uint3)(+0, +1, -1), (uint3)(+1, +1, -1), (uint3)(-1, -1, +0),
    (uint3)(+0, -1, +0), (uint3)(+1, -1, +0), (uint3)(-1, +0, +0), (uint3)(+0, +0, +0), (uint3)(+1, +0, +0),
    (uint3)(-1, +1, +0), (uint3)(+0, +1, +0), (uint3)(+1, +1, +0), (uint3)(-1, -1, +1), (uint3)(+0, -1, +1),
    (uint3)(+1, -1, +1), (uint3)(-1, +0, +1), (uint3)(+0, +0, +1), (uint3)(+1, +0, +1), (uint3)(-1, +1, +1),
    (uint3)(+0, +1, +1), (uint3)(+1, +1, +1)};

const constant float Poly6Factor = 315.f / (64.f * M_PI_F * (HHH * HHH * HHH));
const constant float SpikyKernelFactor = -(45.f / (M_PI_F * HHH * HHH));

inline float poly6Kernel(const float r) { return select(0.f, Poly6Factor * pown(HH - r * r, 3), r <= SPH_H); }

inline float3 spikyKernelGradient(const float3 x, const float3 y, const float r) {
  return (r >= EPSILON && r <= SPH_H) ? (x - y) * (SpikyKernelFactor * native_divide(pown(SPH_H - r, 2), r))
                                      : (float3)(0.f);
}

#define FOR_EACH_NEIGHBOUR__(zIndex, gridTable, gridTableN, op)                                                        \
  {                                                                                                                    \
    const size_t __x = coordAtZCurveGridIndex0((zIndex));                                                              \
    const size_t __y = coordAtZCurveGridIndex1((zIndex));                                                              \
    const size_t __z = coordAtZCurveGridIndex2((zIndex));                                                              \
    size_t __offsets[27] = {                                                                                           \
        zCurveGridIndexAtCoord(__x - 1, __y - 1, __z - 1), zCurveGridIndexAtCoord(__x + 0, __y - 1, __z - 1),          \
        zCurveGridIndexAtCoord(__x + 1, __y - 1, __z - 1), zCurveGridIndexAtCoord(__x - 1, __y + 0, __z - 1),          \
        zCurveGridIndexAtCoord(__x + 0, __y + 0, __z - 1), zCurveGridIndexAtCoord(__x + 1, __y + 0, __z - 1),          \
        zCurveGridIndexAtCoord(__x - 1, __y + 1, __z - 1), zCurveGridIndexAtCoord(__x + 0, __y + 1, __z - 1),          \
        zCurveGridIndexAtCoord(__x + 1, __y + 1, __z - 1), zCurveGridIndexAtCoord(__x - 1, __y - 1, __z + 0),          \
        zCurveGridIndexAtCoord(__x + 0, __y - 1, __z + 0), zCurveGridIndexAtCoord(__x + 1, __y - 1, __z + 0),          \
        zCurveGridIndexAtCoord(__x - 1, __y + 0, __z + 0), zCurveGridIndexAtCoord(__x + 0, __y + 0, __z + 0),          \
        zCurveGridIndexAtCoord(__x + 1, __y + 0, __z + 0), zCurveGridIndexAtCoord(__x - 1, __y + 1, __z + 0),          \
        zCurveGridIndexAtCoord(__x + 0, __y + 1, __z + 0), zCurveGridIndexAtCoord(__x + 1, __y + 1, __z + 0),          \
        zCurveGridIndexAtCoord(__x - 1, __y - 1, __z + 1), zCurveGridIndexAtCoord(__x + 0, __y - 1, __z + 1),          \
        zCurveGridIndexAtCoord(__x + 1, __y - 1, __z + 1), zCurveGridIndexAtCoord(__x - 1, __y + 0, __z + 1),          \
        zCurveGridIndexAtCoord(__x + 0, __y + 0, __z + 1), zCurveGridIndexAtCoord(__x + 1, __y + 0, __z + 1),          \
        zCurveGridIndexAtCoord(__x - 1, __y + 1, __z + 1), zCurveGridIndexAtCoord(__x + 0, __y + 1, __z + 1),          \
        zCurveGridIndexAtCoord(__x + 1, __y + 1, __z + 1)};                                                            \
    for (size_t __i = 0; __i < 27; __i++) {                                                                            \
      const size_t __offset = (__offsets[__i]);                                                                        \
      if (__offset > gridTableN) continue;                                                                             \
      const size_t __start = (gridTable)[__offset];                                                                    \
      const size_t __end = ((__offset + 1) < (gridTableN)) ? (gridTable)[__offset + 1] : (__start);                    \
      for (size_t b = __start; b < __end; ++b) {                                                                       \
        op;                                                                                                            \
      }                                                                                                                \
    }                                                                                                                  \
  }

kernel void check_size(global size_t *sizes) { sizes[get_global_id(0)] = _SIZES[get_global_id(0)]; }

kernel void sph_diffuse(const constant ClSphConfig *config, //
                        const global uint *zIndex,          //
                        const global uint *gridTable,       //
                        const uint gridTableN,              //
                        const global ClSphType *type,       //
                        const global float3 *pStar,         //
                        const global float4 *colour,        //
                        global float4 *diffused) {

  const size_t a = get_global_id(0);
  if (type[a] == Obstacle) return;

  int N = 0;
  float4 mixture = (float4)(0);
  FOR_EACH_NEIGHBOUR__(zIndex[a], gridTable, gridTableN, {
    if (type[b] != Obstacle) {
      mixture += colour[b];
      N++;
    }
  });
  if (N != 0) {
    float4 out = mix(colour[a], (mixture / N) * 1.33f, config->dt / 750.f);
    diffused[a] = clamp(out, 0.03f, 1.f);
  } else {
    diffused[a] = colour[a];
  }
}

kernel void sph_lambda(const constant ClSphConfig *config, //
                       const global uint *zIndex,          //
                       const global uint *gridTable,       //
                       const uint gridTableN,              //
                       const global ClSphType *type,       //
                       const global float3 *pStar,         //
                       const global float *mass,           //
                       global float *lambda) {

  const size_t a = get_global_id(0);
  if (type[a] == Obstacle) {
    lambda[a] = 0;
    return;
  }

  float3 norm2V = (float3)(0.f);
  float rho = 0.f;

  FOR_EACH_NEIGHBOUR__(zIndex[a], gridTable, gridTableN, {
    const float r = fast_distance(pStar[a], pStar[b]);
    norm2V = mad(spikyKernelGradient(pStar[a], pStar[b], r), RHO_RECIP, norm2V);
    rho = mad(mass[a], poly6Kernel(r), rho);
  });
  float norm2 = (norm2V.x * norm2V.x) +
                (norm2V.y * norm2V.y) +
                (norm2V.z * norm2V.z);
  float Ci = (rho / RHO - 1.f);
  lambda[a] = -Ci / (norm2 + CFM_EPSILON);
}

kernel void sph_delta(const constant ClSphConfig *config, //
                      const global uint *zIndex,          //
                      const global uint *gridTable,       //
                      const uint gridTableN,              //
                      const global ClSphType *type,       //
                      global float3 *pStar,               //
                      const global float *lambda,         //
                      const global float3 *position,      //
                      global float3 *velocity,            //
                      global float3 *deltaP

) {
  const size_t a = get_global_id(0);
  if (type[a] == Obstacle) return;

  const float p6DeltaQ = poly6Kernel(CorrDeltaQH);

  float3 deltaPacc = (float3)(0.f);

  FOR_EACH_NEIGHBOUR__(zIndex[a], gridTable, gridTableN, {
    const float r = fast_distance(pStar[a], pStar[b]);
    const float corr = -CorrK * pow(poly6Kernel(r) / p6DeltaQ, CorrN);
    const float factor = (lambda[a] + lambda[b] + corr) / RHO;
    deltaPacc = mad(spikyKernelGradient(pStar[a], pStar[b], r), factor, deltaPacc);
  });

  deltaP[a] = deltaPacc;

  float3 pos = (pStar[a] + deltaP[a]) * config->scale;
  // clamp to extent
  pos = min(config->maxBound, max(config->minBound, pos));
  pStar[a] = pos / config->scale;

#ifdef DEBUG
  printf("[%ld] config { scale=%f, dt=%f} p={id=%ld, mass=%f lam=%f, deltaP=(%f,%f,%f)}\n", id, config.scale, config.dt,
         a->particle.id, a->particle.mass, a->lambda, a->deltaP.x, a->deltaP.y, a->deltaP.z);
#endif
}

kernel void sph_finalise(const constant ClSphConfig *config, //
                         const global ClSphType *type,       //
                         const global float3 *pStar,         //
                         global float3 *position,            //
                         global float3 *velocity) {
  const size_t a = get_global_id(0);
  if (type[a] == Obstacle) return;
  const float3 deltaX = pStar[a] - position[a] / config->scale;
  position[a] = pStar[a] * config->scale;
  velocity[a] = mad(deltaX, (1.f / config->dt), velocity[a]) * VD;
}

kernel void mc_lattice(const constant ClSphConfig *config,  //
                       const constant ClMcConfig *mcConfig, //
                       const global uint *gridTable,        //
                       uint gridTableN,                     //
                       const global ClSphType *type,        //
                       const float3 min,                    //
                       const uint3 sizes,                   //
                       const uint3 gridExtent,              //
                       const global float3 *position,       //
                       const global float4 *colours,        //
                       global float4 *latticePNs,           //
                       global float4 *latticeCs) {

  const size_t x = get_global_id(0);
  const size_t y = get_global_id(1);
  const size_t z = get_global_id(2);

  const float3 pos = (float3)(x, y, z);
  const float step = SPH_H / mcConfig->sampleResolution;
  const float3 a = (min + (pos * step)) * config->scale;

  const size_t zIndex =
      zCurveGridIndexAtCoord((size_t)(pos.x / mcConfig->sampleResolution), (size_t)(pos.y / mcConfig->sampleResolution),
                             (size_t)(pos.z / mcConfig->sampleResolution));

  const float threshold = SPH_H * config->scale * 1;

  const size_t __x = coordAtZCurveGridIndex0(zIndex);
  const size_t __y = coordAtZCurveGridIndex1(zIndex);
  const size_t __z = coordAtZCurveGridIndex2(zIndex);

  if (__x == gridExtent.x && __y == gridExtent.y && __z == gridExtent.z) {
    // XXX there is exactly one case where this may happens: the last element of the z-curve
    return;
  }

  const size_t x_l = clamp(((int)__x) - 1, 0, (int)gridExtent.x - 1);
  const size_t x_r = clamp(((int)__x) + 1, 0, (int)gridExtent.x - 1);
  const size_t y_l = clamp(((int)__y) - 1, 0, (int)gridExtent.y - 1);
  const size_t y_r = clamp(((int)__y) + 1, 0, (int)gridExtent.y - 1);
  const size_t z_l = clamp(((int)__z) - 1, 0, (int)gridExtent.z - 1);
  const size_t z_r = clamp(((int)__z) + 1, 0, (int)gridExtent.z - 1);

  size_t offsets[27] = {zCurveGridIndexAtCoord(x_l, y_l, z_l), zCurveGridIndexAtCoord(__x, y_l, z_l),
                        zCurveGridIndexAtCoord(x_r, y_l, z_l), zCurveGridIndexAtCoord(x_l, __y, z_l),
                        zCurveGridIndexAtCoord(__x, __y, z_l), zCurveGridIndexAtCoord(x_r, __y, z_l),
                        zCurveGridIndexAtCoord(x_l, y_r, z_l), zCurveGridIndexAtCoord(__x, y_r, z_l),
                        zCurveGridIndexAtCoord(x_r, y_r, z_l), zCurveGridIndexAtCoord(x_l, y_l, __z),
                        zCurveGridIndexAtCoord(__x, y_l, __z), zCurveGridIndexAtCoord(x_r, y_l, __z),
                        zCurveGridIndexAtCoord(x_l, __y, __z), zCurveGridIndexAtCoord(__x, __y, __z),
                        zCurveGridIndexAtCoord(x_r, __y, __z), zCurveGridIndexAtCoord(x_l, y_r, __z),
                        zCurveGridIndexAtCoord(__x, y_r, __z), zCurveGridIndexAtCoord(x_r, y_r, __z),
                        zCurveGridIndexAtCoord(x_l, y_l, z_r), zCurveGridIndexAtCoord(__x, y_l, z_r),
                        zCurveGridIndexAtCoord(x_r, y_l, z_r), zCurveGridIndexAtCoord(x_l, __y, z_r),
                        zCurveGridIndexAtCoord(__x, __y, z_r), zCurveGridIndexAtCoord(x_r, __y, z_r),
                        zCurveGridIndexAtCoord(x_l, y_r, z_r), zCurveGridIndexAtCoord(__x, y_r, z_r),
                        zCurveGridIndexAtCoord(x_r, y_r, z_r)};

  float v = 0.f;
  float3 normal = (float3)(0);
  float4 colour = (float4)(0);
  size_t N = 0;
  for (size_t i = 0; i < 27; i++) {
    const size_t offset = offsets[i];
    const size_t start = gridTable[offset];
    const size_t end = (offset + 1 < gridTableN) ? gridTable[offset + 1] : start;
    for (size_t b = start; b < end; ++b) {
      if (type[b] != Obstacle && fast_distance(position[b], a) < threshold) {
        const float3 l = (position[b]) - a;
        const float len = fast_length(l);
        const float denominator = pow(len, mcConfig->particleInfluence);

        normal += (-mcConfig->particleInfluence) * mcConfig->particleSize * (l / denominator);
        v += (mcConfig->particleSize / denominator);
        colour += colours[b];
        N++;
      }
    }
  }
  normal = fast_normalize(normal);

  const size_t idx = index3d(x, y, z, sizes.x, sizes.y, sizes.z);
  latticePNs[idx].s0 = v;
  latticePNs[idx].s1 = normal.s0;
  latticePNs[idx].s2 = normal.s1;
  latticePNs[idx].s3 = normal.s2;
  latticeCs[idx] = colour / N;
}

inline uint3 to3d(size_t id, uint3 pos) {
  return (uint3)(to3dX(id, pos.x, pos.y, pos.z), to3dY(id, pos.x, pos.y, pos.z), to3dZ(id, pos.x, pos.y, pos.z));
}

const constant uint3 CUBE_OFFSETS[8] = {(uint3)(0, 0, 0), (uint3)(1, 0, 0), (uint3)(1, 1, 0), (uint3)(0, 1, 0),
                                        (uint3)(0, 0, 1), (uint3)(1, 0, 1), (uint3)(1, 1, 1), (uint3)(0, 1, 1)};

kernel void mc_size(const constant ClMcConfig *mcConfig, //
                    const uint3 sizes,                   //
                    const global float4 *values,         //
                    local uint *localSums,               //
                    global uint *partialSums) {

  const uint3 marchRange = sizes - (uint3)(1);

  uint nVert = 0;
  // because global size needs to be divisible by local group size (CL1.2), we discard the padding
  if (get_global_id(0) >= (marchRange.x * marchRange.y * marchRange.z)) {
    // NOOP
  } else {
    const uint3 pos = to3d(get_global_id(0), marchRange);
    const float isolevel = mcConfig->isolevel;
    uint ci = 0u;

    for (int i = 0; i < 8; ++i) {
      const uint3 offset = CUBE_OFFSETS[i] + pos;
      const float v = values[index3d(offset.x, offset.y, offset.z, sizes.x, sizes.y, sizes.z)].s0;
      ci = select(ci, ci | (1 << i), v < isolevel);
    }
    nVert = select((uint)NumVertsTable[ci] / 3, 0u, EdgeTable[ci] == 0);
  }

  const uint localId = get_local_id(0);
  const uint groupSize = get_local_size(0);

  // zero out local memory first, this is needed because workgroup size might not divide
  // group size perfectly; we need to zero out trailing cells
  if (localId == 0) {
    for (size_t i = 0; i < get_local_size(0); ++i)
      localSums[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  localSums[localId] = nVert;

  for (uint stride = groupSize / 2; stride > 0; stride >>= 1u) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId < stride) localSums[localId] += localSums[localId + stride];
  }

  if (localId == 0) {
    partialSums[get_group_id(0)] = localSums[0];
  }
}

inline float scale(float isolevel, float v1, float v2) {
  // TODO replace with mix
  return (isolevel - v1) / (v2 - v1);
}

inline void lerpAll(size_t index, float3 *ts, float3 *ns, float4 *cs,                    //
                    size_t from, size_t to,                                              //
                    const float3 *offsets, const float3 *normals, const float4 *colours, //
                    float isolevel, float v0, float v1) {                                //
  const float t = scale(isolevel, v0, v1);
  ts[index] = mix(offsets[from], offsets[to], t);
  ns[index] = mix(normals[from], normals[to], t);
  cs[index] = mix(colours[from], colours[to], t);
}

// FIXME nvidia GPUs output broken triangles for some reason, Intel and AMD works fine
kernel void mc_eval(const constant ClSphConfig *config,  //
                    const constant ClMcConfig *mcConfig, //
                    const float3 min,                    //
                    const uint3 sizes,                   //
                    const global float4 *latticePNs,     //
                    const global float4 *latticeCs,      //
                    volatile global uint *trigCounter,   //
                    const uint acc,                      //
                    global float3 *outVxs, global float3 *outNxs, global float4 *outCxs) {

  const uint3 pos = to3d(get_global_id(0), sizes - (uint3)(1, 1, 1));
  const float isolevel = mcConfig->isolevel;
  const float step = SPH_H / mcConfig->sampleResolution;

  float values[8];
  float3 offsets[8];
  float3 normals[8];
  float4 colours[8];

  uint ci = 0;
  for (int i = 0; i < 8; ++i) {
    const uint3 offset = CUBE_OFFSETS[i] + pos;
    const size_t idx = index3d(offset.x, offset.y, offset.z, sizes.x, sizes.y, sizes.z);

    const float4 point = latticePNs[idx];

    values[i] = point.s0;
    offsets[i] = (min + (convert_float3(offset) * step)) * config->scale;
    normals[i] = (float3)(point.s1, point.s2, point.s3);
    colours[i] = latticeCs[idx];

    ci = select(ci, ci | (1 << i), values[i] < isolevel);
  }

  float3 ts[12];
  float3 ns[12];
  float4 cs[12];

  const uint edge = EdgeTable[ci];

  if (edge & 1 << 0) lerpAll(0, ts, ns, cs, 0, 1, offsets, normals, colours, isolevel, values[0], values[1]);
  if (edge & 1 << 1) lerpAll(1, ts, ns, cs, 1, 2, offsets, normals, colours, isolevel, values[1], values[2]);
  if (edge & 1 << 2) lerpAll(2, ts, ns, cs, 2, 3, offsets, normals, colours, isolevel, values[2], values[3]);
  if (edge & 1 << 3) lerpAll(3, ts, ns, cs, 3, 0, offsets, normals, colours, isolevel, values[3], values[0]);
  if (edge & 1 << 4) lerpAll(4, ts, ns, cs, 4, 5, offsets, normals, colours, isolevel, values[4], values[5]);
  if (edge & 1 << 5) lerpAll(5, ts, ns, cs, 5, 6, offsets, normals, colours, isolevel, values[5], values[6]);
  if (edge & 1 << 6) lerpAll(6, ts, ns, cs, 6, 7, offsets, normals, colours, isolevel, values[6], values[7]);
  if (edge & 1 << 7) lerpAll(7, ts, ns, cs, 7, 4, offsets, normals, colours, isolevel, values[7], values[4]);
  if (edge & 1 << 8) lerpAll(8, ts, ns, cs, 0, 4, offsets, normals, colours, isolevel, values[0], values[4]);
  if (edge & 1 << 9) lerpAll(9, ts, ns, cs, 1, 5, offsets, normals, colours, isolevel, values[1], values[5]);
  if (edge & 1 << 10) lerpAll(10, ts, ns, cs, 2, 6, offsets, normals, colours, isolevel, values[2], values[6]);
  if (edge & 1 << 11) lerpAll(11, ts, ns, cs, 3, 7, offsets, normals, colours, isolevel, values[3], values[7]);

  for (size_t i = 0; TriTable[ci][i] != 255; i += 3) {
    const uint trigIndex = atomic_inc(trigCounter);
    const int x = TriTable[ci][i + 0];
    const int y = TriTable[ci][i + 1];
    const int z = TriTable[ci][i + 2];

    outVxs[trigIndex * 3 + 0] = ts[x];
    outVxs[trigIndex * 3 + 1] = ts[y];
    outVxs[trigIndex * 3 + 2] = ts[z];

    outNxs[trigIndex * 3 + 0] = ns[x];
    outNxs[trigIndex * 3 + 1] = ns[y];
    outNxs[trigIndex * 3 + 2] = ns[z];

    outCxs[trigIndex * 3 + 0] = cs[x];
    outCxs[trigIndex * 3 + 1] = cs[y];
    outCxs[trigIndex * 3 + 2] = cs[z];
  }
  //	printf("trigIdx: %d -> %d", (*trigCounter), trigIndex);
}
