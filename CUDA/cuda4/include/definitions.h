#ifndef DEFINITIONS_H
#define DEFINITIONS_H


#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16


#define WIDTH (BLOCKDIM_X*40)
#define HEIGHT (BLOCKDIM_Y*30)
#define HEIGHT_OFFSET (BLOCKDIM_Y*10)




typedef struct Params
{
  float EPS = 0.001f;
  float step_size = 1.f;
  float min_distance = 0.001f;
  float p[17];
} _Params;

#endif