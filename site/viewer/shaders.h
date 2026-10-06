// Physically based shading for the replay viewer. WebGL 2 (GLSL ES 3.00) on
// the web, GLSL 3.30 natively. Cook-Torrance GGX direct light with a soft
// shadow map, a fill light, and an analytic studio environment (palette
// gradient plus softboxes) for diffuse and specular ambient, ACES tone mapping.
#pragma once

#ifdef PLATFORM_WEB
#define NI_GLSL "#version 300 es\nprecision highp float;\n"
#else
#define NI_GLSL "#version 330\n"
#endif

static const char* ni_vs = NI_GLSL
    "in vec3 vertexPosition;\n"
    "in vec3 vertexNormal;\n"
    "in vec2 vertexTexCoord;\n"
    "uniform mat4 mvp;\n"
    "uniform mat4 matModel;\n"
    "out vec3 worldPos;\n"
    "out vec3 worldNormal;\n"
    "void main(){\n"
    "  vec4 w = matModel*vec4(vertexPosition, 1.0);\n"
    "  worldPos = w.xyz;\n"
    "  worldNormal = mat3(matModel)*vertexNormal;\n"
    "  gl_Position = mvp*vec4(vertexPosition, 1.0) + vec4(0.0*vertexTexCoord, 0.0, 0.0);\n"
    "}\n";

// Studio environment shared by ambient and reflections: palette sky from slate
// (#646DA0) overhead to rose (#D38AAA) at the horizon, a dark floor, and two
// rectangular softboxes that give metals readable highlights.
#define NI_ENV \
    "vec3 lin(vec3 c){return pow(c, vec3(2.2));}\n" \
    "vec3 envColor(vec3 d, float rough){\n" \
    "  float up = clamp(d.z*0.5+0.5, 0.0, 1.0);\n" \
    "  vec3 sky = mix(lin(vec3(0.83,0.54,0.67)), lin(vec3(0.39,0.43,0.63)), smoothstep(0.45, 1.0, up));\n" \
    "  vec3 floorc = lin(vec3(0.34,0.32,0.42));\n" \
    "  vec3 c = mix(floorc, sky, smoothstep(0.30, 0.62, up))*1.25;\n" \
    "  float s = 1.0-rough*0.85;\n" \
    "  vec3 a = normalize(vec3(-0.55,-0.65,0.52)); vec3 b = normalize(vec3(0.70,0.30,0.64));\n" \
    "  c += vec3(4.2,3.9,4.1)*pow(max(dot(d,a),0.0), mix(6.0,160.0,s))*s;\n" \
    "  c += lin(vec3(0.83,0.54,0.67))*3.0*pow(max(dot(d,b),0.0), mix(4.0,90.0,s))*s;\n" \
    "  return c;\n" \
    "}\n"

static const char* ni_fs = NI_GLSL
    "in vec3 worldPos;\n"
    "in vec3 worldNormal;\n"
    "out vec4 finalColor;\n"
    "uniform vec3 albedo;\n"
    "uniform vec4 surface;\n"  // metallic, roughness, emissive, wrap
    "uniform vec3 camPos;\n"
    "uniform vec3 lightDir;\n"  // toward the key light
    "uniform vec3 lightColor;\n"
    "uniform vec3 fillDir;\n"
    "uniform vec3 fillColor;\n"
    "uniform float exposure;\n"
    "uniform sampler2D shadowMap;\n"
    "uniform mat4 lightVP;\n"
    "uniform float texel;\n"
    "uniform vec2 shadowBias;\n"  // normal offset in metres, depth bias in light depth units
    "uniform float opacity;\n"  // 1 for solids; below 1 for glass and see-through tissue (blended pass)
    NI_ENV
    "float shadowAt(vec3 p, vec3 n){\n"
    "  vec4 q = lightVP*vec4(p + n*shadowBias.x, 1.0);\n"
    "  vec3 s = q.xyz/q.w*0.5+0.5;\n"
    "  if(any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0))) || s.z > 1.0) return 1.0;\n"
    "  float bias = shadowBias.y*(1.0 + 2.0*(1.0-max(dot(n, lightDir), 0.0)));\n"
    "  float v = 0.0;\n"
    "  for(int x=-2; x<=2; x++) for(int y=-2; y<=2; y++){\n"
    "    vec2 e = texture(shadowMap, s.xy + vec2(float(x), float(y))*texel).rg;\n"
    "    v += (s.z - bias <= e.r + e.g/255.0) ? 1.0 : 0.0;\n"
    "  }\n"
    "  return v/25.0;\n"
    "}\n"
    "vec3 aces(vec3 x){return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0);}\n"
    "vec2 envBRDF(float r, float nv){\n"
    "  vec4 c0 = vec4(-1.0,-0.0275,-0.572,0.022), c1 = vec4(1.0,0.0425,1.04,-0.04);\n"
    "  vec4 q = r*c0 + c1; float a = min(q.x*q.x, exp2(-9.28*nv))*q.x + q.y;\n"
    "  return vec2(-1.04, 1.04)*a + q.zw;\n"
    "}\n"
    "vec3 direct(vec3 n, vec3 v, vec3 l, vec3 c, vec3 base, float metal, float rough, float wrap){\n"
    "  vec3 h = normalize(v+l);\n"
    "  float nl = max(dot(n,l), 0.0), nv = max(dot(n,v), 1e-4), nh = max(dot(n,h), 0.0), vh = max(dot(v,h), 0.0);\n"
    "  float a = rough*rough, a2 = a*a, den = nh*nh*(a2-1.0)+1.0;\n"
    "  float D = a2/(3.14159*den*den);\n"
    "  float k = (rough+1.0)*(rough+1.0)/8.0;\n"
    "  float G = (nl/(nl*(1.0-k)+k))*(nv/(nv*(1.0-k)+k));\n"
    "  vec3 f0 = mix(vec3(0.04), base, metal);\n"
    "  vec3 F = f0 + (1.0-f0)*pow(1.0-vh, 5.0);\n"
    "  vec3 spec = D*G*F/max(4.0*nl*nv, 1e-4);\n"
    "  float diffuseTerm = max((dot(n,l)+wrap)/(1.0+wrap), 0.0);\n"
    "  vec3 diff = (1.0-F)*(1.0-metal)*base/3.14159*diffuseTerm;\n"
    "  return (diff + spec*nl)*c;\n"
    "}\n"
    "void main(){\n"
    "  vec3 n = normalize(worldNormal);\n"
    "  if(!gl_FrontFacing) n = -n;\n"
    "  vec3 v = normalize(camPos - worldPos);\n"
    "  float metal = surface.x, rough = clamp(surface.y, 0.04, 1.0), emissive = surface.z, wrap = surface.w;\n"
    "  float sh = shadowAt(worldPos, n);\n"
    "  vec3 color = direct(n, v, lightDir, lightColor*sh, albedo, metal, rough, wrap);\n"
    "  color += direct(n, v, fillDir, fillColor, albedo, metal, rough, wrap);\n"
    "  float nv = max(dot(n,v), 1e-4);\n"
    "  vec3 f0 = mix(vec3(0.04), albedo, metal);\n"
    "  vec2 ab = envBRDF(rough, nv);\n"
    "  vec3 irradiance = envColor(n, 1.0);\n"
    "  vec3 reflection = envColor(reflect(-v, n), rough);\n"
    "  float occlusion = mix(0.55, 1.0, sh);\n"
    "  color += (1.0-metal)*albedo*irradiance*0.55*occlusion;\n"
    "  color += reflection*(f0*ab.x + ab.y)*mix(0.6, 1.0, sh);\n"
    "  color += albedo*emissive;\n"
    "  color = aces(color*exposure);\n"
    "  finalColor = vec4(pow(color, vec3(1.0/2.2)), opacity);\n"
    "}\n";

// Depth for the shadow pass, packed into two 8-bit channels (portable in WebGL 2).
static const char* ni_depth_fs = NI_GLSL
    "in vec3 worldPos;\n"
    "in vec3 worldNormal;\n"
    "out vec4 finalColor;\n"
    "void main(){\n"
    "  float d = gl_FragCoord.z;\n"
    "  float hi = floor(d*255.0)/255.0;\n"
    "  finalColor = vec4(hi, (d-hi)*255.0, 0.0, 1.0);\n"
    "}\n";
