// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Created by S.Guillitte
void main() {

  vec2 uv = (gl_FragCoord.xy - iResolution.xy) / iResolution.y;

  vec3 col = vec3(0);
  vec3 rd = normalize(vec3(uv.x, uv.y, 1));
  
  
  gl_FragColor = vec4(col, 1.0);
}

sudo apt install cmake
sudo apt install xorg-dev
mkdir build 
cmake -S . -B build 
cd build
make 
