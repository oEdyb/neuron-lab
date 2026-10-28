export const vertex = `
attribute float strength;
attribute float rank;
varying float power;
varying float visibility;
uniform float fade;
uniform float focus;
uniform vec2 viewport;
uniform vec2 reveal;
uniform vec2 sparseRange;
uniform float pixelRatio;
uniform float glow;
void main(){
  vec4 p=modelViewMatrix*vec4(position,1.);
  vec4 clip=projectionMatrix*p;
  vec4 px=projectionMatrix*(p+modelViewMatrix*vec4(.147,0.,0.,0.));
  vec4 pz=projectionMatrix*(p+modelViewMatrix*vec4(0.,0.,.143,0.));
  float spacing=min(length((px.xy/px.w-clip.xy/clip.w)*viewport*.5),length((pz.xy/pz.w-clip.xy/clip.w)*viewport*.5));
  float grid=smoothstep(reveal.x,reveal.y,spacing);
  float sparse=smoothstep(sparseRange.x,sparseRange.y,rank);
  power=clamp(log2(1.+strength)/5.,.05,1.);
  visibility=(grid*.16+sparse*(.35+power*.65))*fade;
  gl_Position=clip;
  gl_PointSize=(mix(2.5,glow,power)*mix(.75,1.,focus))*pixelRatio;
}`;
export const fragment = `
varying float power;
varying float visibility;
uniform vec3 tint;
void main(){
  float d=length(gl_PointCoord-.5)*2.;
  if(d>1.)discard;
  float core=exp(-d*d*65.);
  float halo=exp(-d*d*6.)*.27;
  float alpha=(core+halo)*visibility*(1.-smoothstep(.75,1.,d));
  gl_FragColor=vec4(mix(tint,vec3(1.,.96,.87),power),alpha);
}`;
