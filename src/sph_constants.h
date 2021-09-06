#pragma once

#include "cl_types.h"

static __device_constant float VD = 0.49f;    // Velocity dampening
static __device_constant float RHO = 6378.0f; // Reference density
static __device_constant float RHO_RECIP = 1.f / RHO;

static __device_constant float EPSILON = 0.00000001f;
static __device_constant float CFM_EPSILON = 600.0f; // CFM propagation
static __device_constant float CorrDeltaQ = 0.3f;

static __device_constant float C = 0.00001f;
static __device_constant float VORTICITY_EPSILON = 0.0005f;
static __device_constant float CorrK = 0.0001f;
static __device_constant float CorrN = 4.f;
