#shader vertex
#version 330 core
layout(location = 0) in vec4 position;
void main()
{
   gl_Position = position;
}

#shader fragment
#version 330 core

// uniform vec4 u_Color;
// uniform float u_Time;

// void main()
// {
//    gl_FragColor = u_Color * u_Time;
// }

uniform vec3 ta;
uniform vec3 ro;

const int   max_iter      = 120;
const vec3  bone          = vec3(0.89, 0.855, 0.788);

float step_size = 0.5;
float min_distance = 0.001 * 0.5;

// F R A C T A L   P A R A M S
float fixed_radius2 = 1.9;
float min_radius2 = 0.1;
float folding_limit = 1.0;
float scale = -2.8;

void sphere_fold(inout vec3 z, inout float dz) {
    float r2 = dot(z, z);
    if (r2 < min_radius2) {
        float temp = (fixed_radius2 / min_radius2);
        z *= temp;
        dz *= temp;
    } else if (r2 < fixed_radius2) {
        float temp = (fixed_radius2 / r2);
        z *= temp;
        dz *= temp;
    }
}

void box_fold(inout vec3 z, inout float dz) {
    z = clamp(z, -folding_limit, folding_limit) * 2.0 - z;
}

float f(vec3 z) { // ray march
    vec3 offset = z;
    float dr = 1.0;
    for (int n = 0; n < 15; ++n) {
        box_fold(z, dr);
        sphere_fold(z, dr);

        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0;
    }
    float r = length(z);
    return r / abs(dr);
}

float intersect(vec3 ro, vec3 rd, out int iter) {
  float res;
  float t = 0.0;
  iter = max_iter;
    
  for(int i = 0; i < max_iter; ++i) {
    vec3 p = ro + rd * t;
    res = f(p);
    if(res < min_distance * t || res > 20.) {
      iter = i;
      break;
    }
    t += res * step_size;
  }
    
  if(res > 20.) t = -1.;
  return t;
}

float ambientOcclusion(vec3 p, vec3 n) {
  float stepSize = 0.012;
  float t = stepSize;

  float oc = 0.0;

  for(int i = 0; i < 12; i++) {
    float d = f(p + n * t);
    oc += t - d;
    t += stepSize;
  }

  return clamp(oc, 0.0, 1.0);
}

vec3 normal(in vec3 pos) {
  vec3  eps = vec3(.001,0.0,0.0);
  vec3 nor;
  nor.x = f(pos+eps.xyy) - f(pos-eps.xyy);
  nor.y = f(pos+eps.yxy) - f(pos-eps.yxy);
  nor.z = f(pos+eps.yyx) - f(pos-eps.yyx);
  return normalize(nor);
}

vec3 lighting(vec3 p, vec3 rd, int iter) {
  vec3 n = normal(p);
  float fake = float(iter)/float(max_iter);
  float fakeAmb = exp(-fake*fake*9.0);
  float amb = ambientOcclusion(p, n);

  vec3 col = vec3(mix(1.0, 0.125, pow(amb, 3.0)))*vec3(fakeAmb)*bone;
  return col;
}

void main() {
    vec2 iResolution = vec2(1000.0,1000.0); // gl_FragCoord is pixel number from 0 to 599
    vec2 q = gl_FragCoord.xy / iResolution.xy; // now q is 0..1 from top/bot and L/R
    vec2 uv = -1.0 + 2.0 * q; // now uv becomes -1 to 1 from top/bot and L/R
    uv.x *= iResolution.x / iResolution.y; // now wide screen has x axis -2..2 


    // send ray origin & lookat
    //vec3 ta = vec3(0.0, 0.0, 0.0);
    //vec3 ro = vec3(0.0, 2.0, 5.9);

    // get the coordinate axes directions
    vec3 cf = normalize(ta - ro);
    vec3 cs = normalize(cross(cf, vec3(0.0, 1.0, 0.0)));
    vec3 cu = normalize(cross(cs, cf));
    vec3 rd = normalize(uv.x * cs + uv.y * cu + 2.8 * cf); // transform from view to world

    // do the intersection
    vec3 bg = mix(bone*0.5, bone, smoothstep(-1.0, 1.0, uv.y));
    vec3 col = bg;
    vec3 p = ro;
    int iter = 0;
    float t = intersect(ro, rd, iter);
    if (t > -0.5) {
        p = ro + t * rd;
        col = lighting(p, rd, iter);
        col = mix(col, bg, 1.0-exp(-0.001*t*t)); 
    }

    gl_FragColor = vec4(col.x, col.y, col.z, 1.0);
}












