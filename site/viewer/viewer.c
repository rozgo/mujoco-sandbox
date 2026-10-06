// Interactive replay of recorded surgical-robot episodes (raylib 6).
// Replays recorded MuJoCo states exported by sixlegs.neural_insertion.site_export;
// it does not simulate. Display interpolates linearly between recorded 50 Hz states.
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PLATFORM_WEB
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

#include "shaders.h"

#define MAX_MOVING 8
#define SHADOW_SIZE 4096

typedef struct {
    char name[32];
    Vector3 pos;
    Quaternion quat;
    int moving;
} Body;

typedef struct {
    char name[16];
    float albedo[3];
    float surface[4];  // metallic, roughness, emissive, wrap
} Mat;

typedef struct {
    int body, mat;
    Mesh mesh;
} Chunk;

typedef struct {
    unsigned seed, target, policy, outcome, frames;
    float goal[3];
    float* data;  // frames x stride: t, moving positions, tip, lateral um, vertical um
} Episode;

static struct {
    Body* bodies;
    int nbody;
    Mat* mats;
    int nmat;
    Chunk* chunks;
    int nchunk;
    Episode* episodes;
    int nepisode, nmoving, stride;
    int moving[MAX_MOVING];
    int selected;
    float time, speed, hold;
    int playing, autoplay;
    Shader pbr, depth;
    Material material, depthMaterial;
    RenderTexture2D shadow;
    int loc[16];
    Matrix lightVP;
    Camera3D camera;
    float yaw, pitch, distance;
    Vector3 target;
    int follow;
    Vector3 current[MAX_MOVING];
    Vector3 tip;
    float lateral, vertical;
} V;

enum { L_ALBEDO, L_SURFACE, L_CAMPOS, L_LIGHTDIR, L_LIGHTCOLOR, L_FILLDIR, L_FILLCOLOR, L_EXPOSURE, L_LIGHTVP, L_TEXEL, L_BIAS };

static const Vector3 LIGHT_DIR = {-0.42f, -0.58f, 0.70f};
static const Vector3 FIELD = {0.0f, -0.055f, 0.112f};

// ---------------------------------------------------------------- loading

typedef struct {
    unsigned char* data;
    int size, at;
} Reader;

static void need(Reader* r, int n) {
    if (r->at + n > r->size) {
        fprintf(stderr, "viewer: truncated data file\n");
        exit(2);
    }
}

static void take(Reader* r, void* out, int n) {
    need(r, n);
    memcpy(out, r->data + r->at, n);
    r->at += n;
}

static unsigned u32(Reader* r) { unsigned v; take(r, &v, 4); return v; }
static int i32(Reader* r) { int v; take(r, &v, 4); return v; }
static float f32(Reader* r) { float v; take(r, &v, 4); return v; }

static Reader open_file(const char* path, const char* magic) {
    Reader r = {0};
    r.data = LoadFileData(path, &r.size);
    if (!r.data || r.size < 4 || memcmp(r.data, magic, 4) != 0) {
        fprintf(stderr, "viewer: cannot read %s\n", path);
        exit(2);
    }
    r.at = 4;
    return r;
}

static void load_scene(const char* path) {
    Reader r = open_file(path, "NIS1");
    V.nbody = (int)u32(&r);
    V.nmat = (int)u32(&r);
    V.nchunk = (int)u32(&r);
    V.bodies = calloc(V.nbody, sizeof(Body));
    for (int b = 0; b < V.nbody; b++) {
        i32(&r);
        take(&r, V.bodies[b].name, 32);
        V.bodies[b].pos = (Vector3){f32(&r), f32(&r), f32(&r)};
        float w = f32(&r), x = f32(&r), y = f32(&r), z = f32(&r);  // MuJoCo w, x, y, z
        V.bodies[b].quat = (Quaternion){x, y, z, w};
        unsigned char moving;
        take(&r, &moving, 1);
        V.bodies[b].moving = moving;
    }
    V.mats = calloc(V.nmat, sizeof(Mat));
    for (int m = 0; m < V.nmat; m++) {
        take(&r, V.mats[m].name, 16);
        for (int k = 0; k < 3; k++) V.mats[m].albedo[k] = f32(&r);
        for (int k = 0; k < 4; k++) V.mats[m].surface[k] = f32(&r);
    }
    V.chunks = calloc(V.nchunk, sizeof(Chunk));
    for (int c = 0; c < V.nchunk; c++) {
        Chunk* ch = &V.chunks[c];
        ch->body = i32(&r);
        ch->mat = i32(&r);
        int nv = (int)u32(&r), ni = (int)u32(&r);
        Mesh mesh = {0};
        mesh.vertexCount = nv;
        mesh.triangleCount = ni / 3;
        mesh.vertices = MemAlloc(nv * 3 * sizeof(float));
        mesh.normals = MemAlloc(nv * 3 * sizeof(float));
        mesh.texcoords = MemAlloc(nv * 2 * sizeof(float));  // bound by the non-VAO path
        mesh.indices = MemAlloc(ni * sizeof(unsigned short));
        take(&r, mesh.vertices, nv * 12);
        take(&r, mesh.normals, nv * 12);
        take(&r, mesh.indices, ni * 2);
        UploadMesh(&mesh, false);
        ch->mesh = mesh;
    }
    UnloadFileData(r.data);
}

static void load_replays(const char* path) {
    Reader r = open_file(path, "NIR1");
    V.nepisode = (int)u32(&r);
    V.nmoving = (int)u32(&r);
    if (V.nmoving > MAX_MOVING) exit(2);
    for (int k = 0; k < V.nmoving; k++) V.moving[k] = i32(&r);
    V.stride = 1 + 3 * V.nmoving + 3 + 2;
    V.episodes = calloc(V.nepisode, sizeof(Episode));
    for (int e = 0; e < V.nepisode; e++) {
        Episode* ep = &V.episodes[e];
        ep->seed = u32(&r);
        ep->target = u32(&r);
        ep->policy = u32(&r);
        ep->outcome = u32(&r);
        ep->frames = u32(&r);
        for (int k = 0; k < 3; k++) ep->goal[k] = f32(&r);
        ep->data = malloc(sizeof(float) * V.stride * ep->frames);
        take(&r, ep->data, sizeof(float) * V.stride * ep->frames);
    }
    UnloadFileData(r.data);
}

// ---------------------------------------------------------------- playback

static Episode* episode(void) { return &V.episodes[V.selected]; }

static float duration(void) {
    Episode* e = episode();
    return e->data[(e->frames - 1) * V.stride];
}

static void sample(float t) {
    Episode* e = episode();
    int k = 0;
    while (k + 1 < (int)e->frames && e->data[(k + 1) * V.stride] <= t) k++;
    int j = k + 1 < (int)e->frames ? k + 1 : k;
    float t0 = e->data[k * V.stride], t1 = e->data[j * V.stride];
    float a = (j > k && t1 > t0) ? Clamp((t - t0) / (t1 - t0), 0, 1) : 0;
    const float* p = e->data + k * V.stride;
    const float* q = e->data + j * V.stride;
    for (int m = 0; m < V.nmoving; m++) {
        V.current[m] = Vector3Lerp((Vector3){p[1 + 3 * m], p[2 + 3 * m], p[3 + 3 * m]},
                                   (Vector3){q[1 + 3 * m], q[2 + 3 * m], q[3 + 3 * m]}, a);
    }
    int o = 1 + 3 * V.nmoving;
    V.tip = Vector3Lerp((Vector3){p[o], p[o + 1], p[o + 2]}, (Vector3){q[o], q[o + 1], q[o + 2]}, a);
    // Error readouts come from the nearest recorded state, not the interpolation.
    const float* n = a < 0.5f ? p : q;
    V.lateral = n[o + 3];
    V.vertical = n[o + 4];
}

static Matrix body_transform(int b) {
    Vector3 pos = V.bodies[b].pos;
    for (int m = 0; m < V.nmoving; m++) {
        if (V.moving[m] == b) pos = V.current[m];
    }
    return MatrixMultiply(QuaternionToMatrix(V.bodies[b].quat), MatrixTranslate(pos.x, pos.y, pos.z));
}

// ---------------------------------------------------------------- camera

static void camera_preset(int preset) {
    V.follow = preset == 2;
    if (preset == 0) {
        V.target = (Vector3){0.0f, -0.02f, 0.22f}; V.distance = 1.05f; V.yaw = -2.25f; V.pitch = 0.36f;
    } else if (preset == 1) {
        V.target = FIELD; V.distance = 0.20f; V.yaw = -2.05f; V.pitch = 0.55f;
    } else {
        V.target = V.tip; V.distance = 0.045f; V.yaw = -2.05f; V.pitch = 0.22f;
    }
}

static void update_camera(void) {
    Vector2 delta = GetMouseDelta();
    int pan = IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && IsKeyDown(KEY_LEFT_SHIFT));
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !pan) {
        V.yaw -= delta.x * 0.006f;
        V.pitch = Clamp(V.pitch + delta.y * 0.006f, -0.2f, 1.45f);
    }
    Vector3 forward = {cosf(V.pitch) * cosf(V.yaw), cosf(V.pitch) * sinf(V.yaw), sinf(V.pitch)};
    if (pan) {
        Vector3 right = Vector3Normalize(Vector3CrossProduct((Vector3){0, 0, 1}, forward));
        Vector3 up = Vector3CrossProduct(forward, right);
        float s = V.distance * 0.0016f;
        V.target = Vector3Add(V.target, Vector3Add(Vector3Scale(right, delta.x * s), Vector3Scale(up, delta.y * s)));
        V.follow = 0;
    }
    float wheel = GetMouseWheelMove();
    if (wheel != 0) V.distance = Clamp(V.distance * expf(-wheel * 0.12f), 0.006f, 3.0f);
    if (V.follow) V.target = V.tip;
    V.camera.target = V.target;
    V.camera.position = Vector3Add(V.target, Vector3Scale(forward, V.distance));
    V.camera.up = (Vector3){0, 0, 1};
    V.camera.fovy = 40.0f;
    V.camera.projection = CAMERA_PERSPECTIVE;
}

// ---------------------------------------------------------------- drawing

static void set_vec3(int loc, Vector3 v) { SetShaderValue(V.pbr, V.loc[loc], &v, SHADER_UNIFORM_VEC3); }

static void draw_chunks(Material material, int shade) {
    for (int c = 0; c < V.nchunk; c++) {
        Chunk* ch = &V.chunks[c];
        if (shade) {
            Mat* m = &V.mats[ch->mat];
            SetShaderValue(V.pbr, V.loc[L_ALBEDO], m->albedo, SHADER_UNIFORM_VEC3);
            SetShaderValue(V.pbr, V.loc[L_SURFACE], m->surface, SHADER_UNIFORM_VEC4);
        }
        DrawMesh(ch->mesh, material, body_transform(ch->body));
    }
}

static void shadow_pass(void) {
    Vector3 dir = Vector3Normalize(LIGHT_DIR);
    Camera3D light = {0};
    light.target = (Vector3){0.0f, -0.03f, 0.13f};
    light.position = Vector3Add(light.target, Vector3Scale(dir, 1.0f));
    light.up = (Vector3){0, 0, 1};
    light.fovy = 0.62f;
    light.projection = CAMERA_ORTHOGRAPHIC;
    rlSetClipPlanes(0.2, 2.0);
    BeginTextureMode(V.shadow);
    ClearBackground(WHITE);
    BeginMode3D(light);
    V.lightVP = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());
    draw_chunks(V.depthMaterial, 0);
    EndMode3D();
    EndTextureMode();
}

static void draw_trail(void) {
    Episode* e = episode();
    int o = 1 + 3 * V.nmoving;
    Color trail = (Color){211, 138, 170, 255};
    for (int k = 0; k + 1 < (int)e->frames && e->data[(k + 1) * V.stride] <= V.time; k++) {
        const float* p = e->data + k * V.stride;
        const float* q = e->data + (k + 1) * V.stride;
        DrawLine3D((Vector3){p[o], p[o + 1], p[o + 2]}, (Vector3){q[o], q[o + 1], q[o + 2]}, trail);
    }
    Vector3 goal = {e->goal[0], e->goal[1], e->goal[2]};
    Color mark = (Color){236, 232, 242, 255};
    float s = 0.0008f;
    DrawLine3D(Vector3Add(goal, (Vector3){-s, 0, 0}), Vector3Add(goal, (Vector3){s, 0, 0}), mark);
    DrawLine3D(Vector3Add(goal, (Vector3){0, -s, 0}), Vector3Add(goal, (Vector3){0, s, 0}), mark);
    DrawLine3D(goal, Vector3Add(goal, (Vector3){0, 0, -0.00085f}), mark);
}

static void draw_frame(void) {
    shadow_pass();
    Vector3 dir = Vector3Normalize(LIGHT_DIR);
    Vector3 fill = Vector3Normalize((Vector3){0.65f, 0.35f, 0.45f});
    Vector3 lightColor = {2.7f, 2.55f, 2.5f}, fillColor = {0.75f, 0.68f, 1.05f};
    float exposure = 0.82f, texel = 1.0f / SHADOW_SIZE;
    float bias[2] = {0.00025f, 0.00005f / 1.8f};
    BeginDrawing();
    int w = GetScreenWidth(), h = GetScreenHeight();
    DrawRectangleGradientV(0, 0, w, h / 2, (Color){100, 109, 160, 255}, (Color){137, 129, 175, 255});
    DrawRectangleGradientV(0, h / 2, w, h - h / 2, (Color){137, 129, 175, 255}, (Color){211, 138, 170, 255});
    rlSetClipPlanes(fmaxf(V.distance * 0.02f, 0.0004f), 8.0);
    BeginMode3D(V.camera);
    set_vec3(L_CAMPOS, V.camera.position);
    set_vec3(L_LIGHTDIR, dir);
    set_vec3(L_LIGHTCOLOR, lightColor);
    set_vec3(L_FILLDIR, fill);
    set_vec3(L_FILLCOLOR, fillColor);
    SetShaderValue(V.pbr, V.loc[L_EXPOSURE], &exposure, SHADER_UNIFORM_FLOAT);
    SetShaderValue(V.pbr, V.loc[L_TEXEL], &texel, SHADER_UNIFORM_FLOAT);
    SetShaderValue(V.pbr, V.loc[L_BIAS], bias, SHADER_UNIFORM_VEC2);
    SetShaderValueMatrix(V.pbr, V.loc[L_LIGHTVP], V.lightVP);
    draw_chunks(V.material, 1);
    draw_trail();
    EndMode3D();
    EndDrawing();
}

static void step(void) {
    float dt = GetFrameTime();
    if (dt > 0.1f) dt = 0.1f;
    if (V.playing) {
        if (V.time < duration()) {
            V.time = fminf(V.time + dt * V.speed, duration());
        } else if ((V.hold += dt) > 1.6f) {
            V.hold = 0;
            V.time = 0;
            if (V.autoplay) {
                // Advance to the next episode of the same policy.
                int policy = (int)episode()->policy;
                for (int k = 1; k <= V.nepisode; k++) {
                    int c = (V.selected + k) % V.nepisode;
                    if ((int)V.episodes[c].policy == policy) {
                        V.selected = c;
                        break;
                    }
                }
            }
        }
    }
    sample(V.time);
    update_camera();
    draw_frame();
}

// ---------------------------------------------------------------- page API

EXPORT int ni_episode_count(void) { return V.nepisode; }
EXPORT unsigned ni_seed(int i) { return V.episodes[i].seed; }
EXPORT int ni_policy(int i) { return (int)V.episodes[i].policy; }
EXPORT int ni_target(int i) { return (int)V.episodes[i].target; }
EXPORT int ni_outcome(int i) { return (int)V.episodes[i].outcome; }
EXPORT int ni_selected(void) { return V.selected; }
EXPORT void ni_select(int i) { if (i >= 0 && i < V.nepisode) { V.selected = i; V.time = 0; V.hold = 0; } }
EXPORT float ni_time(void) { return V.time; }
EXPORT float ni_duration(void) { return duration(); }
EXPORT void ni_seek(float t) { V.time = Clamp(t, 0, duration()); V.hold = 0; }
EXPORT void ni_play(int on) { V.playing = on; }
EXPORT int ni_playing(void) { return V.playing; }
EXPORT void ni_speed(float s) { V.speed = s; }
EXPORT void ni_autoplay(int on) { V.autoplay = on; }
EXPORT void ni_camera(int preset) { camera_preset(preset); }
EXPORT float ni_lateral(void) { return V.lateral; }
EXPORT float ni_vertical(void) { return V.vertical; }
EXPORT void ni_resize(int w, int h) { SetWindowSize(w, h); }

// ---------------------------------------------------------------- startup

static void init(const char* folder, int width, int height) {
#ifdef PLATFORM_WEB
    SetConfigFlags(FLAG_MSAA_4X_HINT);  // the page sizes the canvas through ni_resize
#else
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
#endif
    SetTraceLogLevel(LOG_WARNING);
    InitWindow(width, height, "Surgical robot replay");
    char path[512];
    snprintf(path, sizeof(path), "%s/scene.bin", folder);
    load_scene(path);
    snprintf(path, sizeof(path), "%s/replays.bin", folder);
    load_replays(path);
    V.pbr = LoadShaderFromMemory(ni_vs, ni_fs);
    V.depth = LoadShaderFromMemory(ni_vs, ni_depth_fs);
    const char* names[] = {"albedo", "surface", "camPos", "lightDir", "lightColor", "fillDir", "fillColor",
                           "exposure", "lightVP", "texel", "shadowBias"};
    for (int k = 0; k <= L_BIAS; k++) V.loc[k] = GetShaderLocation(V.pbr, names[k]);
    V.shadow = LoadRenderTexture(SHADOW_SIZE, SHADOW_SIZE);
    if (!IsRenderTextureValid(V.shadow)) {
        fprintf(stderr, "viewer: shadow framebuffer unavailable\n");
        exit(2);
    }
    SetTextureFilter(V.shadow.texture, TEXTURE_FILTER_POINT);
    SetTextureWrap(V.shadow.texture, TEXTURE_WRAP_CLAMP);
    V.material = LoadMaterialDefault();
    V.material.shader = V.pbr;
    V.material.maps[MATERIAL_MAP_ALBEDO].texture = V.shadow.texture;  // sampled as texture0
    V.depthMaterial = LoadMaterialDefault();
    V.depthMaterial.shader = V.depth;
    V.speed = 1.0f;
    V.playing = 1;
    V.autoplay = 1;
    sample(0);
    camera_preset(1);
    update_camera();
}

#ifdef PLATFORM_WEB
int main(void) {
    init("data", 1280, 720);
    emscripten_set_main_loop(step, 0, 1);
    return 0;
}
#else
// Native: interactive window, or --screenshot for scripted inspection.
int main(int argc, char** argv) {
    const char* folder = "data";
    const char* shot = NULL;
    int width = 1600, height = 900, select = 0, camera = 1;
    float at = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) folder = argv[++i];
        else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc) shot = argv[++i];
        else if (!strcmp(argv[i], "--episode") && i + 1 < argc) select = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i + 1 < argc) at = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--camera") && i + 1 < argc) camera = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 2 < argc) { width = atoi(argv[++i]); height = atoi(argv[++i]); }
    }
    init(folder, width, height);
    ni_select(select);
    if (at >= 0) { ni_seek(at); ni_play(0); }
    sample(V.time);
    camera_preset(camera);
    if (shot) {
        for (int f = 0; f < 3; f++) { update_camera(); draw_frame(); }
        Image image = LoadImageFromScreen();
        ExportImage(image, shot);
        UnloadImage(image);
        CloseWindow();
        return 0;
    }
    SetTargetFPS(60);
    while (!WindowShouldClose()) step();
    CloseWindow();
    return 0;
}
#endif
