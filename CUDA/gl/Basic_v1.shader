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

float intersect(vec3 ro, vec3 rd) {
    float res;
    float t = 0.0;
    for (int i = 0; i < 128; ++i) {
        vec3 p = ro + rd * t;
        res = f(p);
        if (res < 0.001 * t || res > 20.)
            break;
        t += res;
    }

    if (res > 20.) t = -1.;
    return t;
}

float softshadow(vec3 ro, vec3 rd, float k) {
    float akuma = 1.0, h = 0.0;
    float t = 0.01;
    for (int i = 0; i < 50; ++i) {
        h = f(ro + rd * t);
        if (h < 0.001) return 0.02;
        akuma = min(akuma, k * h / t);
        t += clamp(h, 0.01, 2.0);
    }
    return akuma;
}

vec3 lighting(vec3 p, vec3 rd, float ps) {

    vec3 l1_dir = normalize(vec3(0.8, 0.8, 0.4));
    vec3 l1_col = 0.3 * vec3(1.5, 1.69, 0.79);
    vec3 l2_dir = normalize(vec3(-0.8, 0.5, 0.3));
    vec3 l2_col = vec3(0.89, 0.99, 1.3);

    vec3 e = vec3(0.5 * ps, 0.0, 0.0);
    vec3 n = normalize(vec3(f(p + e.xyy) - f(p - e.xyy),
        f(p + e.yxy) - f(p - e.yxy),
        f(p + e.yyx) - f(p - e.yyx)));

    float shadow = softshadow(p, l1_dir, 10.0);

    float dif1 = max(0.0, dot(n, l1_dir));
    float dif2 = max(0.0, dot(n, l2_dir));
    float bac1 = max(0.3 + 0.7 * dot(vec3(-l1_dir.x, -1.0, -l1_dir.z), n), 0.0);
    float bac2 = max(0.2 + 0.8 * dot(vec3(-l2_dir.x, -1.0, -l2_dir.z), n), 0.0);
    float spe = max(0.0, pow(clamp(dot(l1_dir, reflect(rd, n)), 0.0, 1.0), 10.0));

    vec3 col = 5.5 * l1_col * dif1 * shadow;
    col += 1.1 * l2_col * dif2;
    col += 0.3 * bac1 * l1_col;
    col += 0.3 * bac2 * l2_col;
    col += 1.0 * spe;
    return col;
}

void main() {
    vec2 iResolution = vec2(600.0,600.0); // gl_FragCoord is pixel number from 0 to 599
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
    vec3 p = ro;
    vec3 col = vec3(1.0);
    float t = intersect(ro, rd);
    if (t > -0.5) {
        p = ro + t * rd;
        col = lighting(p, rd, 0.004) * vec3(1.0, 1.3, 1.23) * 0.4;
    }
    gl_FragColor = vec4(col.x, col.y, col.z, 1.0);
    //gl_FragColor = vec4(gl_FragCoord.x / iResolution.x, gl_FragCoord.y / iResolution.x,0.0,0.0);
}