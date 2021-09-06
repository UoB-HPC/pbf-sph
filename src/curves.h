#pragma once

#ifndef __OPENCL_C_VERSION__
#include <cstddef>
#endif

//#if !defined(CURVE_UINT3_TYPE) || !defined(CURVE_UINT3_CTOR)
//#error type/ctor of 3 component vector not defined
//#endif

#ifndef __cplusplus
#define _constexpr const
#else
#define _constexpr constexpr
#endif

_constexpr size_t index3d(size_t x, size_t y, size_t z, size_t xMax, size_t yMax, size_t zMax) {
  return x * yMax * zMax + y * zMax + z;
}

_constexpr size_t to3dX(size_t index, size_t xMax, size_t yMax, size_t zMax) {
  size_t x = index / (yMax * zMax);
  return x;
}

_constexpr size_t to3dY(size_t index, size_t xMax, size_t yMax, size_t zMax) {
  size_t x = index / (yMax * zMax);
  size_t y = (index - x * yMax * zMax) / zMax;
  return y;
}

_constexpr size_t to3dZ(size_t index, size_t xMax, size_t yMax, size_t zMax) {
  size_t x = index / (yMax * zMax);
  size_t y = (index - x * yMax * zMax) / zMax;
  size_t z = index - x * yMax * zMax - y * zMax;
  return z;
}

//inline CURVE_UINT3_TYPE to3d(size_t index, size_t xMax, size_t yMax, size_t zMax) {
//  size_t x = index / (yMax * zMax);
//  size_t y = (index - x * yMax * zMax) / zMax;
//  size_t z = index - x * yMax * zMax - y * zMax;
//  return CURVE_UINT3_CTOR(x, y, z);
//}

_constexpr size_t uninterleave(size_t value) {
  size_t ret = 0x0;
  ret |= (value & 0x1) >> 0;
  ret |= (value & 0x8) >> 2;
  ret |= (value & 0x40) >> 4;
  ret |= (value & 0x200) >> 6;
  ret |= (value & 0x1000) >> 8;
  ret |= (value & 0x8000) >> 10;
  ret |= (value & 0x40000) >> 12;
  ret |= (value & 0x200000) >> 14;
  ret |= (value & 0x1000000) >> 16;
  ret |= (value & 0x8000000) >> 18;
  return ret;
}

_constexpr size_t coordAtZCurveGridIndex0(size_t index) { return uninterleave((index)&0x9249249); }

_constexpr size_t coordAtZCurveGridIndex1(size_t index) { return uninterleave((index >> 1) & 0x9249249); }

_constexpr size_t coordAtZCurveGridIndex2(size_t index) { return uninterleave((index >> 2) & 0x9249249); }

//inline CURVE_UINT3_TYPE coordAtZCurveGridIndex(size_t zIndex) {
//  return CURVE_UINT3_CTOR(coordAtZCurveGridIndex0(zIndex), coordAtZCurveGridIndex1(zIndex),
//                          coordAtZCurveGridIndex2(zIndex));
//}

_constexpr size_t zCurveGridIndexAtCoord(size_t x, size_t y, size_t z) {
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8)) & 0x0300F00F;
  x = (x | (x << 4)) & 0x030C30C3;
  x = (x | (x << 2)) & 0x09249249;

  y = (y | (y << 16)) & 0x030000FF;
  y = (y | (y << 8)) & 0x0300F00F;
  y = (y | (y << 4)) & 0x030C30C3;
  y = (y | (y << 2)) & 0x09249249;

  z = (z | (z << 16)) & 0x030000FF;
  z = (z | (z << 8)) & 0x0300F00F;
  z = (z | (z << 4)) & 0x030C30C3;
  z = (z | (z << 2)) & 0x09249249;
  return x | y << 1 | z << 2;
}

#undef _constexpr