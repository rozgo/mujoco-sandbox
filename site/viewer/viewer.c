// Interactive replay of recorded surgical-robot episodes (raylib 6).
// Replays recorded MuJoCo states exported by sixlegs.neural_insertion.site_export;
// it does not simulate. Display interpolates linearly between recorded 50 Hz states.
// Overlays draw the recorded disturbance components at stated, fixed scales:
// slide forces as arrows along each joint axis, table acceleration as an arrow and
// a trace, and the measured tip, target estimate and moving true target at true
// scale (visible in the micro camera).
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <math.h>
#include <stdarg.h>
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
#define NJ 5
#define FORCE_SCALE 0.50f     // m of arrow per N (5 cm per 0.1 N)
#define ACC_SCALE 2.0f        // m of arrow per m/s^2 (2 cm per 10 mm/s^2)
#define MAX_CALLOUTS 12
#define CALLOUT_W 560         // callout texture size, pixels
#define CALLOUT_H 200
#define CALLOUT_FONT 30.0f
#define TOLERANCE 1e-5f       // success tolerance, m (lateral and vertical)

enum { OV_FORCES = 1, OV_SENSING = 2, OV_HUD = 4 };

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
    float level, latency;
    float goal[3];  // mean true goal over the episode (camera anchor)
    float* data;    // frames x stride, see the F_ offsets
} Episode;

// Frame layout after time, moving-body positions and tip (site_export.EXTRA).
enum { X_LAT, X_VERT, X_GOAL = 2, X_MTIP = 5, X_MGOAL = 8, X_ACC = 11, X_FORCE = 14, X_FRICTION = 19,
       X_INERTIAL = 24, X_ACTION = 29, X_COUNT = 30 };

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
    // Disturbance overlays.
    int extra, overlays, camPreset;
    Vector3 axis[MAX_MOVING], centroid[MAX_MOVING];
    float x[X_COUNT];  // interpolated extra fields at the display time
    Font font, bold;
    RenderTexture2D callout[MAX_CALLOUTS];
} V;

// A data callout: a camera-facing panel placed in the scene near the object it
// describes, joined to it by a leader line. Panels are sized in world units
// proportional to the orbit distance, so they keep perspective and parallax.
typedef struct {
    Vector3 anchor;  // object of interest
    float dx, dy;    // panel offset: camera-right and screen-up, in orbit distances
    int lines;       // line 0 is the title
    char label[5][48], value[5][32];
    Color color[5];
    float w, h;      // panel size in texture pixels, fixed per callout layout
} Callout;

static Callout C[MAX_CALLOUTS];
static int NC;

static const Color C_NOISE = {255, 180, 94, 255};     // force noise
static const Color C_FRICTION = {111, 211, 224, 255}; // extra Coulomb friction
static const Color C_VIBRATION = {242, 238, 255, 255};// table vibration
static const Color C_MEASURED = {126, 232, 181, 255}; // delayed, noisy tip measurement
static const Color C_ESTIMATE = {255, 159, 200, 255}; // target estimate
static const Color C_TRUTH = {255, 255, 255, 255};    // true moving target
static const Color C_TIP = {211, 138, 170, 255};

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
    Reader r = open_file(path, "NIR2");
    V.nepisode = (int)u32(&r);
    V.nmoving = (int)u32(&r);
    V.extra = (int)u32(&r);
    if (V.nmoving > MAX_MOVING || V.extra != X_COUNT) {
        fprintf(stderr, "viewer: unexpected replay layout\n");
        exit(2);
    }
    for (int k = 0; k < V.nmoving; k++) V.moving[k] = i32(&r);
    for (int k = 0; k < V.nmoving; k++) V.axis[k] = (Vector3){f32(&r), f32(&r), f32(&r)};
    V.stride = 1 + 3 * V.nmoving + 3 + V.extra;
    V.episodes = calloc(V.nepisode, sizeof(Episode));
    int o = 1 + 3 * V.nmoving + 3;
    for (int e = 0; e < V.nepisode; e++) {
        Episode* ep = &V.episodes[e];
        ep->seed = u32(&r);
        ep->target = u32(&r);
        ep->policy = u32(&r);
        ep->outcome = u32(&r);
        ep->frames = u32(&r);
        ep->level = f32(&r);
        ep->latency = f32(&r);
        ep->data = malloc(sizeof(float) * V.stride * ep->frames);
        take(&r, ep->data, sizeof(float) * V.stride * ep->frames);
        for (int k = 0; k < 3; k++) {
            double sum = 0;
            for (unsigned f = 0; f < ep->frames; f++) sum += ep->data[f * V.stride + o + X_GOAL + k];
            ep->goal[k] = (float)(sum / ep->frames);
        }
    }
    UnloadFileData(r.data);
    // Arrow anchors: centroid of each moving body's geometry, in body coordinates.
    for (int m = 0; m < V.nmoving; m++) {
        double sum[3] = {0, 0, 0};
        long n = 0;
        for (int c = 0; c < V.nchunk; c++) {
            if (V.chunks[c].body != V.moving[m]) continue;
            Mesh* mesh = &V.chunks[c].mesh;
            for (int v = 0; v < mesh->vertexCount; v++, n++) {
                for (int k = 0; k < 3; k++) sum[k] += mesh->vertices[3 * v + k];
            }
        }
        V.centroid[m] = n ? (Vector3){(float)(sum[0] / n), (float)(sum[1] / n), (float)(sum[2] / n)} : Vector3Zero();
    }
}

// ---------------------------------------------------------------- playback

EXPORT void ni_select(int i);

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
    for (int k = 0; k < X_COUNT; k++) V.x[k] = p[o + 3 + k] + a * (q[o + 3 + k] - p[o + 3 + k]);
    // Error readouts come from the nearest recorded state, not the interpolation.
    const float* n = a < 0.5f ? p : q;
    V.lateral = n[o + 3 + X_LAT];
    V.vertical = n[o + 3 + X_VERT];
}

static Vector3 xvec(int k) { return (Vector3){V.x[k], V.x[k + 1], V.x[k + 2]}; }

static Vector3 frame_vec(const Episode* e, int f, int k) {
    const float* p = e->data + f * V.stride + 1 + 3 * V.nmoving + 3 + k;
    return (Vector3){p[0], p[1], p[2]};
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
    V.camPreset = preset;
    if (preset == 0) {
        V.target = (Vector3){0.0f, -0.02f, 0.22f}; V.distance = 1.05f; V.yaw = -2.25f; V.pitch = 0.36f;
    } else if (preset == 1) {
        V.target = FIELD; V.distance = 0.20f; V.yaw = -2.05f; V.pitch = 0.55f;
    } else if (preset == 2) {
        V.target = V.tip; V.distance = 0.045f; V.yaw = -2.05f; V.pitch = 0.22f;
    } else {
        // Micro: fixed on the episode's mean target, about 0.5 mm away, true scale.
        Episode* e = episode();
        V.target = (Vector3){e->goal[0], e->goal[1], e->goal[2]}; V.distance = 0.0005f; V.yaw = -2.05f; V.pitch = 0.16f;
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
    if (wheel != 0) V.distance = Clamp(V.distance * expf(-wheel * 0.12f), 0.0003f, 3.0f);
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
    if (V.overlays & OV_SENSING) return;  // the overlay draws the moving target instead
    Vector3 goal = xvec(X_GOAL);
    Color mark = (Color){236, 232, 242, 255};
    float s = 0.0008f;
    DrawLine3D(Vector3Add(goal, (Vector3){-s, 0, 0}), Vector3Add(goal, (Vector3){s, 0, 0}), mark);
    DrawLine3D(Vector3Add(goal, (Vector3){0, -s, 0}), Vector3Add(goal, (Vector3){0, s, 0}), mark);
    DrawLine3D(goal, Vector3Add(goal, (Vector3){0, 0, -0.00085f}), mark);
}

// ---------------------------------------------------------------- overlays

static float px(void) { return V.distance * 0.0012f; }  // world size of ~1 screen pixel at 720p

static void arrow(Vector3 from, Vector3 vec, float radius, Color c) {
    float len = Vector3Length(vec);
    if (len < radius) return;
    Vector3 dir = Vector3Scale(vec, 1.0f / len);
    float head = fminf(radius * 5.0f, len * 0.5f);
    Vector3 neck = Vector3Add(from, Vector3Scale(dir, len - head));
    DrawCylinderEx(from, neck, radius, radius, 10, c);
    DrawCylinderEx(neck, Vector3Add(from, vec), radius * 2.4f, 0.0f, 12, c);
}

static void cross(Vector3 p, float s, Color c) {
    DrawLine3D(Vector3Add(p, (Vector3){-s, 0, 0}), Vector3Add(p, (Vector3){s, 0, 0}), c);
    DrawLine3D(Vector3Add(p, (Vector3){0, -s, 0}), Vector3Add(p, (Vector3){0, s, 0}), c);
    DrawLine3D(Vector3Add(p, (Vector3){0, 0, -s}), Vector3Add(p, (Vector3){0, 0, s}), c);
}

static void dashed(Vector3 a, Vector3 b, float dash, Color c) {
    float len = Vector3Distance(a, b);
    int n = (int)fminf(len / dash, 400);
    if (n < 2) { DrawLine3D(a, b, c); return; }
    for (int k = 0; k < n; k += 2) {
        DrawLine3D(Vector3Lerp(a, b, (float)k / n), Vector3Lerp(a, b, fminf((float)(k + 1) / n, 1)), c);
    }
}

static Color fade(Color c, float a) { c.a = (unsigned char)(255 * Clamp(a, 0, 1)); return c; }

static Vector3 vibration_anchor(void) { return (Vector3){0.112f, -0.040f, 0.004f}; }

static void draw_forces(void) {
    float r = px() * 1.6f;
    for (int m = 0; m < V.nmoving && m < NJ; m++) {
        Vector3 anchor = Vector3Transform(V.centroid[m], body_transform(V.moving[m]));
        Vector3 ax = V.axis[m];
        Vector3 side = Vector3Normalize(Vector3CrossProduct(ax, fabsf(ax.z) > 0.9f ? (Vector3){1, 0, 0} : (Vector3){0, 0, 1}));
        float comps[3] = {V.x[X_FORCE + m], V.x[X_FRICTION + m], V.x[X_INERTIAL + m]};
        Color colors[3] = {C_NOISE, C_FRICTION, C_VIBRATION};
        for (int k = 0; k < 3; k++) {
            Vector3 from = Vector3Add(anchor, Vector3Scale(side, (k - 1) * r * 4.0f));
            arrow(from, Vector3Scale(ax, comps[k] * FORCE_SCALE), r, colors[k]);
        }
    }
    // Table vibration: acceleration arrow and the last 0.5 s of recorded samples.
    Episode* e = episode();
    Vector3 base = vibration_anchor();
    arrow(base, Vector3Scale(xvec(X_ACC), ACC_SCALE), r * 1.2f, C_VIBRATION);
    int now = 0;
    while (now + 1 < (int)e->frames && e->data[(now + 1) * V.stride] <= V.time) now++;
    for (int f = (now > 15 ? now - 15 : 0); f < now; f++) {
        Vector3 a = Vector3Add(base, Vector3Scale(frame_vec(e, f, X_ACC), ACC_SCALE));
        Vector3 b = Vector3Add(base, Vector3Scale(frame_vec(e, f + 1, X_ACC), ACC_SCALE));
        DrawLine3D(a, b, fade(C_VIBRATION, 0.1f + 0.5f * (f - now + 15) / 15.0f));
    }
    DrawSphere(base, r * 1.5f, C_VIBRATION);
}

static void draw_sensing(void) {
    Episode* e = episode();
    int now = 0;
    while (now + 1 < (int)e->frames && e->data[(now + 1) * V.stride] <= V.time) now++;
    float s = px() * 5.0f;
    Vector3 goal = xvec(X_GOAL), estimate = xvec(X_MGOAL), measured = xvec(X_MTIP);
    // True target path over the last 2 s (breathing and pulse), at true scale.
    for (int f = (now > 100 ? now - 100 : 0); f < now; f++) {
        DrawLine3D(frame_vec(e, f, X_GOAL), frame_vec(e, f + 1, X_GOAL), fade(C_TRUTH, 0.15f + 0.6f * (f - now + 100) / 100.0f));
    }
    // Recent target estimates as a fading cloud: noise, bias and drift.
    for (int f = (now > 25 ? now - 25 : 0); f <= now; f++) {
        cross(frame_vec(e, f, X_MGOAL), s * 0.35f, fade(C_ESTIMATE, 0.2f + 0.6f * (f - now + 25) / 25.0f));
    }
    // Success tolerance around the true target: 10 um lateral, 10 um vertical, true scale.
    for (int k = -1; k <= 1; k += 2) {
        DrawCircle3D(Vector3Add(goal, (Vector3){0, 0, k * TOLERANCE}), TOLERANCE, (Vector3){1, 0, 0}, 0, fade(C_TRUTH, 0.55f));
    }
    for (int k = 0; k < 4; k++) {
        float th = k * PI / 2;
        Vector3 rim = Vector3Add(goal, (Vector3){TOLERANCE * cosf(th), TOLERANCE * sinf(th), 0});
        DrawLine3D(Vector3Add(rim, (Vector3){0, 0, -TOLERANCE}), Vector3Add(rim, (Vector3){0, 0, TOLERANCE}), fade(C_TRUTH, 0.55f));
    }
    DrawSphere(goal, s * 0.45f, C_TRUTH);
    cross(estimate, s, C_ESTIMATE);
    DrawSphereWires(estimate, s * 0.6f, 6, 10, C_ESTIMATE);
    dashed(goal, estimate, s * 0.4f, fade(C_ESTIMATE, 0.8f));
    // What the policy is told about its tip: delayed by the latency and noisy.
    DrawSphereWires(measured, s * 0.6f, 6, 10, C_MEASURED);
    dashed(V.tip, measured, s * 0.4f, fade(C_MEASURED, 0.8f));
    DrawSphere(V.tip, s * 0.45f, C_TIP);
}

static Callout* callout(Vector3 anchor, float dx, float dy, const char* title, Color color) {
    if (NC >= MAX_CALLOUTS) return NULL;
    Callout* c = &C[NC++];
    memset(c, 0, sizeof(*c));
    c->anchor = anchor; c->dx = dx; c->dy = dy;
    snprintf(c->label[0], sizeof(c->label[0]), "%s", title);
    c->color[0] = color;
    c->lines = 1;
    return c;
}

// A row: a fixed label and a value right-aligned in a fixed column. Panels are
// sized from the labels and a worst-case value, never from the current value, so
// they do not resize as numbers change (digits in DejaVu Sans share one width).
static void row(Callout* c, Color color, const char* label, const char* value) {
    if (!c || c->lines >= 5) return;
    snprintf(c->label[c->lines], sizeof(c->label[0]), "%s", label);
    snprintf(c->value[c->lines], sizeof(c->value[0]), "%s", value);
    c->color[c->lines++] = color;
}

// Signed number with a typographic minus (same advance as plus) and fixed decimals.
static const char* num(char* buf, float v, int decimals, int sign) {
    char tmp[32];
    double scale = pow(10, decimals), r = round((double)v * scale) / scale;
    if (r == 0) r = 0;  // no "-0.0"
    snprintf(tmp, sizeof(tmp), sign ? "%+.*f" : "%.*f", decimals, r);
    if (tmp[0] == '-') snprintf(buf, 32, "\xe2\x88\x92%s", tmp + 1);
    else snprintf(buf, 32, "%s", tmp);
    return buf;
}

// Lengths in micrometres below 1 mm, millimetres above, so the digit count stays bounded.
static const char* length(char* buf, float metres, int sign) {
    char n[32];
    if (fabsf(metres) < 1e-3f) snprintf(buf, 32, "%s \xc2\xb5m", num(n, metres * 1e6f, 1, sign));
    else snprintf(buf, 32, "%s mm", num(n, metres * 1e3f, 2, sign));
    return buf;
}

static const Color C_INK = {236, 232, 246, 255};
#define VALUE_TEMPLATE "\xe2\x88\x92" "000.0 \xc2\xb5m"  // widest value any callout shows

static void collect_callouts(void) {
    NC = 0;
    Episode* e = episode();
    float d = V.distance;
    char b[32];
    const char* names[NJ] = {"X slide", "Y slide", "Z slide", "insertion slide", "retainer slide"};
    const float off[NJ][2] = {{0.16f, 0.07f}, {-0.22f, 0.05f}, {0.17f, -0.02f}, {-0.20f, -0.07f}, {0.15f, -0.24f}};
    if ((V.overlays & OV_FORCES) && d > 0.5f) {
        for (int m = 0; m < V.nmoving && m < NJ; m++) {
            Vector3 anchor = Vector3Transform(V.centroid[m], body_transform(V.moving[m]));
            Callout* c = callout(anchor, off[m][0], off[m][1], names[m], WHITE);
            char n[32];
            snprintf(b, sizeof(b), "%s mN", num(n, V.x[X_FORCE + m] * 1e3f, 1, 1));
            row(c, C_NOISE, "noise", b);
            snprintf(b, sizeof(b), "%s mN", num(n, V.x[X_FRICTION + m] * 1e3f, 1, 1));
            row(c, C_FRICTION, "friction", b);
            snprintf(b, sizeof(b), "%s mN", num(n, V.x[X_INERTIAL + m] * 1e3f, 1, 1));
            row(c, C_VIBRATION, "vibration", b);
        }
    }
    if ((V.overlays & OV_FORCES) && d > 0.12f) {
        Vector3 acc = xvec(X_ACC);
        char n[32];
        Callout* c = callout(vibration_anchor(), 0.10f, 0.05f, "table vibration", WHITE);
        snprintf(b, sizeof(b), "%s mm/s\xc2\xb2", num(n, Vector3Length(acc) * 1e3f, 1, 0));
        row(c, C_VIBRATION, "acceleration", b);
        snprintf(b, sizeof(b), "%s mm/s\xc2\xb2", num(n, acc.z * 1e3f, 1, 1));
        row(c, C_VIBRATION, "vertical", b);
    }
    if (!(V.overlays & OV_SENSING)) return;
    Vector3 goal = xvec(X_GOAL), mean = {e->goal[0], e->goal[1], e->goal[2]};
    float meas = Vector3Distance(xvec(X_MTIP), V.tip), est = Vector3Distance(xvec(X_MGOAL), goal);
    float moved = Vector3Distance(goal, mean);
    char late[32];
    snprintf(late, sizeof(late), "%.0f ms", e->latency * 1e3f);
    if (d < 0.01f) {
        Callout* c = callout(V.tip, -0.30f, 0.17f, "needle tip (true)", C_TIP);
        row(c, C_INK, "lateral", length(b, V.lateral * 1e-6f, 0));
        row(c, C_INK, "vertical", length(b, V.vertical * 1e-6f, 1));
        c = callout(xvec(X_MTIP), 0.26f, 0.17f, "measured tip", C_MEASURED);
        row(c, C_INK, "late by", late);
        row(c, C_INK, "off by", length(b, meas, 0));
        c = callout(goal, 0.34f, -0.02f, "true target", WHITE);
        row(c, C_INK, "tissue motion", length(b, moved, 0));
        row(c, C_INK, "tolerance", "\xc2\xb1" "10.0 \xc2\xb5m");
        c = callout(xvec(X_MGOAL), 0.22f, -0.22f, "target estimate", C_ESTIMATE);
        row(c, C_INK, "off by", length(b, est, 0));
        Matrix view = GetCameraMatrix(V.camera);
        Vector3 right = {view.m0, view.m4, view.m8};
        Vector3 ruler = Vector3Add((Vector3){mean.x, mean.y, mean.z + 25e-6f}, Vector3Scale(right, -d * 0.42f));
        callout(ruler, -0.04f, 0.0f, "50 \xc2\xb5m", C_INK);
    } else if (d < 0.5f) {
        Callout* c = callout(V.tip, -0.22f, 0.10f, "needle tip", C_TIP);
        row(c, C_INK, "lateral", length(b, V.lateral * 1e-6f, 0));
        row(c, C_INK, "vertical", length(b, V.vertical * 1e-6f, 1));
        row(c, C_MEASURED, "measured, late by", late);
        row(c, C_MEASURED, "measured, off by", length(b, meas, 0));
        Vector3 site = Vector3Add(goal, (Vector3){0, 0, -0.001f});
        c = callout(site, 0.16f, -0.08f, "target", WHITE);
        row(c, C_INK, "hover height", "1.00 mm");
        row(c, C_INK, "tissue motion", length(b, moved, 0));
        row(c, C_ESTIMATE, "estimate off by", length(b, est, 0));
    }
}

static void render_callouts(void) {
    float tw = MeasureTextEx(V.font, VALUE_TEMPLATE, CALLOUT_FONT, 0).x;
    for (int k = 0; k < NC; k++) {
        Callout* c = &C[k];
        // Width from the title, the (constant) labels and the template value only.
        float w = MeasureTextEx(V.bold, c->label[0], CALLOUT_FONT, 0).x;
        for (int i = 1; i < c->lines; i++) w = fmaxf(w, MeasureTextEx(V.font, c->label[i], CALLOUT_FONT, 0).x + 24 + tw);
        c->w = fminf(w + 30, CALLOUT_W);
        c->h = fminf(c->lines * CALLOUT_FONT * 1.25f + 18, CALLOUT_H);
        BeginTextureMode(V.callout[k]);
        ClearBackground(BLANK);
        DrawRectangleRounded((Rectangle){0, 0, c->w, c->h}, 0.18f, 8, (Color){22, 20, 42, 214});
        DrawRectangleRounded((Rectangle){0, 0, 6, c->h}, 0.5f, 4, c->color[0]);
        for (int i = 0; i < c->lines; i++) {
            float y = 9 + i * CALLOUT_FONT * 1.25f;
            DrawTextEx(i ? V.font : V.bold, c->label[i], (Vector2){16, y}, CALLOUT_FONT, 0, i ? fade(c->color[i], 0.85f) : c->color[0]);
            if (i && c->value[i][0]) {
                float vw = MeasureTextEx(V.font, c->value[i], CALLOUT_FONT, 0).x;
                BeginScissorMode(16, (int)y, (int)(c->w - 30), (int)(CALLOUT_FONT * 1.25f));  // clip, never grow
                DrawTextEx(V.font, c->value[i], (Vector2){c->w - 14 - vw, y}, CALLOUT_FONT, 0, c->color[i]);
                EndScissorMode();
            }
        }
        EndTextureMode();
    }
}

#define CALLOUT_ZOOM_CAP 1.4f  // orbit distance beyond which panels shrink with the scene

static void draw_callouts(void) {
    Matrix view = GetCameraMatrix(V.camera);
    Vector3 right = {view.m0, view.m4, view.m8}, up = {view.m1, view.m5, view.m9};
    float scale = fminf(V.distance, CALLOUT_ZOOM_CAP);
    float fov = tanf(V.camera.fovy * DEG2RAD * 0.5f), sh = (float)GetScreenHeight();
    Vector3 at[MAX_CALLOUTS];
    float w[MAX_CALLOUTS], h[MAX_CALLOUTS], wpp[MAX_CALLOUTS];
    Rectangle r[MAX_CALLOUTS];
    for (int k = 0; k < NC; k++) {
        Callout* c = &C[k];
        at[k] = Vector3Add(c->anchor, Vector3Add(Vector3Scale(right, c->dx * scale), Vector3Scale(up, c->dy * scale)));
        // Partial perspective: distant panels shrink by the square root of the depth ratio.
        float depth = Vector3Distance(V.camera.position, at[k]);
        float unit = sqrtf(scale * fmaxf(depth * scale / V.distance, scale * 0.2f)) * 0.00050f;
        w[k] = c->w * unit;
        h[k] = c->h * unit;
        wpp[k] = 2.0f * depth * fov / sh;  // world size of one screen pixel at the panel
        Vector2 p = GetWorldToScreen(at[k], V.camera);
        float pw = w[k] / wpp[k], ph = h[k] / wpp[k];
        r[k] = (Rectangle){c->dx < 0 ? p.x - pw : p.x, p.y - ph * 0.5f, pw, ph};
    }
    // Keep panels from overlapping on screen: push overlapping pairs apart vertically.
    float shift[MAX_CALLOUTS] = {0};
    for (int pass = 0; pass < 8; pass++) {
        int moved = 0;
        for (int i = 0; i < NC; i++) {
            for (int j = i + 1; j < NC; j++) {
                Rectangle a = r[i], b = r[j];
                a.y += shift[i]; b.y += shift[j];
                float gap = 6.0f;
                if (a.x + a.width + gap <= b.x || b.x + b.width + gap <= a.x) continue;
                if (a.y + a.height + gap <= b.y || b.y + b.height + gap <= a.y) continue;
                int lower = a.y + a.height * 0.5f > b.y + b.height * 0.5f ? i : j, upper = lower == i ? j : i;
                Rectangle L = lower == i ? a : b, U = lower == i ? b : a;
                float overlap = U.y + U.height + gap - L.y;
                shift[lower] += overlap * 0.5f;
                shift[upper] -= overlap * 0.5f;
                moved = 1;
            }
        }
        if (!moved) break;
    }
    for (int k = 0; k < NC; k++) {
        Callout* c = &C[k];
        Vector3 pos = Vector3Subtract(at[k], Vector3Scale(up, shift[k] * wpp[k]));
        // The leader line meets the panel edge nearest the object.
        Vector3 bl = c->dx < 0 ? Vector3Subtract(pos, Vector3Scale(right, w[k])) : pos;
        bl = Vector3Subtract(bl, Vector3Scale(up, h[k] * 0.5f));
        DrawLine3D(c->anchor, pos, fade(c->color[0], 0.9f));
        DrawSphere(c->anchor, scale * 0.0016f, c->color[0]);
        float u = c->w / CALLOUT_W, v = c->h / CALLOUT_H;
        rlSetTexture(V.callout[k].texture.id);
        rlBegin(RL_QUADS);
        rlColor4ub(255, 255, 255, 255);
        rlNormal3f(0, 0, 1);
        Vector3 p1 = Vector3Add(bl, Vector3Scale(right, w[k])), p2 = Vector3Add(p1, Vector3Scale(up, h[k])), p3 = Vector3Add(bl, Vector3Scale(up, h[k]));
        rlTexCoord2f(0, 1 - v); rlVertex3f(bl.x, bl.y, bl.z);
        rlTexCoord2f(u, 1 - v); rlVertex3f(p1.x, p1.y, p1.z);
        rlTexCoord2f(u, 1); rlVertex3f(p2.x, p2.y, p2.z);
        rlTexCoord2f(0, 1); rlVertex3f(p3.x, p3.y, p3.z);
        rlEnd();
        rlSetTexture(0);
    }
}

// A 3D ruler: a labelled vertical scale at true size beside the target.
static void draw_ruler(void) {
    if (!(V.overlays & OV_SENSING) || V.distance >= 0.01f) return;
    Matrix view = GetCameraMatrix(V.camera);
    Vector3 right = {view.m0, view.m4, view.m8};
    Episode* e = episode();
    Vector3 base = Vector3Add((Vector3){e->goal[0], e->goal[1], e->goal[2] - 25e-6f}, Vector3Scale(right, -V.distance * 0.42f));
    // (labelled by a callout at its midpoint)
    Color c = fade(C_INK, 0.8f);
    DrawLine3D(base, Vector3Add(base, (Vector3){0, 0, 50e-6f}), c);
    for (int k = 0; k <= 5; k++) {
        Vector3 t = Vector3Add(base, (Vector3){0, 0, k * 10e-6f});
        DrawLine3D(t, Vector3Add(t, Vector3Scale(right, (k % 5 ? 3e-6f : 6e-6f))), c);
    }
}

static void swatch_row(float x, float y, float w, float size, Color c, const char* name, const char* value) {
    DrawRectangleRounded((Rectangle){x, y + size * 0.22f, size * 0.62f, size * 0.62f}, 0.3f, 4, c);
    DrawTextEx(V.font, name, (Vector2){x + size * 0.95f, y}, size, 0, (Color){236, 232, 246, 255});
    Vector2 m = MeasureTextEx(V.font, value, size, 0);
    DrawTextEx(V.font, value, (Vector2){x + w - m.x, y}, size, 0, WHITE);
}

static void draw_hud(void) {
    Episode* e = episode();
    int w = GetScreenWidth(), h = GetScreenHeight();
    float s = fminf(h / 720.0f, w / 900.0f), size = 14.0f * s, line = size * 1.36f;
    float pw = 270 * s, chart = 84 * s, ph = line * 9.4f + chart + 16 * s;
    float x = 14 * s, y = h - ph - 14 * s;
    DrawRectangleRounded((Rectangle){x, y, pw, ph}, 0.06f, 8, (Color){24, 22, 44, 190});
    x += 12 * s;
    y += 10 * s;
    float iw = pw - 24 * s;
    char buf[64];
    snprintf(buf, sizeof(buf), "DISTURBANCE LEVEL %.2f", e->level);
    DrawTextEx(V.bold, buf, (Vector2){x, y}, size, 0, (Color){255, 214, 232, 255});
    y += line * 1.15f;
    swatch_row(x, y, iw, size, C_NOISE, "slide force noise", ""); y += line;
    swatch_row(x, y, iw, size, C_FRICTION, "extra friction", ""); y += line;
    swatch_row(x, y, iw, size, C_VIBRATION, "table vibration", ""); y += line;
    swatch_row(x, y, iw, size, C_MEASURED, "tip as measured (late, noisy)", ""); y += line;
    swatch_row(x, y, iw, size, C_ESTIMATE, "target as estimated", ""); y += line;
    swatch_row(x, y, iw, size, C_TRUTH, "true target (tissue motion)", ""); y += line;
    DrawTextEx(V.font, "force arrows 5 cm per 0.1 N", (Vector2){x, y}, size * 0.85f, 0, (Color){200, 196, 226, 255});
    y += line * 1.05f;
    DrawTextEx(V.font, "height vs mean target, \xc2\xb1" "40 \xc2\xb5m", (Vector2){x, y}, size * 0.85f, 0, (Color){200, 196, 226, 255});
    y += line;
    Rectangle box = {x, y, iw, chart};
    DrawRectangleRec(box, (Color){12, 11, 26, 150});
    float span = 40e-6f, total = e->data[(e->frames - 1) * V.stride];
    float mid = box.y + box.height * 0.5f;
    Vector3 mean = {e->goal[0], e->goal[1], e->goal[2]};
    DrawLine((int)box.x, (int)mid, (int)(box.x + box.width), (int)mid, (Color){90, 86, 130, 255});
    int fields[3] = {X_GOAL, X_MGOAL, X_MTIP};
    Color colors[4] = {C_TRUTH, C_ESTIMATE, C_MEASURED, C_TIP};
    for (int series = 0; series < 4; series++) {
        Vector2 prev = {0};
        for (unsigned f = 0; f < e->frames; f++) {
            float t = e->data[f * V.stride];
            if (t > V.time) break;
            float z = series < 3 ? frame_vec(e, f, fields[series]).z : e->data[f * V.stride + 1 + 3 * V.nmoving + 2];
            Vector2 p = {box.x + box.width * t / fmaxf(total, 1e-3f),
                         mid - box.height * 0.5f * Clamp((z - mean.z) / span, -1, 1)};
            if (f > 0) DrawLineEx(prev, p, series == 0 ? 2.0f * s : 1.3f * s, fade(colors[series], series == 1 ? 0.7f : 1));
            prev = p;
        }
    }
}

static void draw_overlays_3d(void) {
    rlDrawRenderBatchActive();
    rlDisableDepthTest();
    rlDisableBackfaceCulling();
    if (V.overlays & OV_FORCES) draw_forces();
    if (V.overlays & OV_SENSING) draw_sensing();
    draw_ruler();
    draw_callouts();
    rlDrawRenderBatchActive();
    rlEnableBackfaceCulling();
    rlEnableDepthTest();
}

static void draw_overlays_2d(void) {
    if (V.overlays & OV_HUD) draw_hud();
}

static void draw_frame(void) {
    shadow_pass();
    collect_callouts();
    render_callouts();
    Vector3 dir = Vector3Normalize(LIGHT_DIR);
    Vector3 fill = Vector3Normalize((Vector3){0.65f, 0.35f, 0.45f});
    Vector3 lightColor = {2.7f, 2.55f, 2.5f}, fillColor = {0.75f, 0.68f, 1.05f};
    float exposure = 0.82f, texel = 1.0f / SHADOW_SIZE;
    float bias[2] = {0.00025f, 0.00005f / 1.8f};
    BeginDrawing();
    int w = GetScreenWidth(), h = GetScreenHeight();
    DrawRectangleGradientV(0, 0, w, h / 2, (Color){100, 109, 160, 255}, (Color){137, 129, 175, 255});
    DrawRectangleGradientV(0, h / 2, w, h - h / 2, (Color){137, 129, 175, 255}, (Color){211, 138, 170, 255});
    rlSetClipPlanes(V.distance * 0.05f, Clamp(V.distance * 2000.0f, 1.0f, 8.0f));
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
    draw_overlays_3d();
    EndMode3D();
    draw_overlays_2d();
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
                    if ((int)V.episodes[c].policy == policy && V.episodes[c].level == episode()->level) {
                        ni_select(c);
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
EXPORT void ni_select(int i) {
    if (i >= 0 && i < V.nepisode) {
        V.selected = i; V.time = 0; V.hold = 0;
        if (V.camPreset == 3) camera_preset(3);
    }
}
EXPORT float ni_time(void) { return V.time; }
EXPORT float ni_duration(void) { return duration(); }
EXPORT void ni_seek(float t) { V.time = Clamp(t, 0, duration()); V.hold = 0; }
EXPORT void ni_play(int on) { V.playing = on; }
EXPORT int ni_playing(void) { return V.playing; }
EXPORT void ni_speed(float s) { V.speed = s; }
EXPORT void ni_autoplay(int on) { V.autoplay = on; }
EXPORT void ni_camera(int preset) { camera_preset(preset); }
EXPORT int ni_camera_preset(void) { return V.camPreset; }
EXPORT float ni_lateral(void) { return V.lateral; }
EXPORT float ni_vertical(void) { return V.vertical; }
EXPORT void ni_resize(int w, int h) { SetWindowSize(w, h); }
EXPORT float ni_level(int i) { return V.episodes[i].level; }
EXPORT float ni_latency(int i) { return V.episodes[i].latency; }
EXPORT void ni_overlays(int mask) { V.overlays = mask; }
EXPORT int ni_overlay_mask(void) { return V.overlays; }

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
    int codepoints[100], n = 0;
    for (int c = 32; c < 127; c++) codepoints[n++] = c;
    codepoints[n++] = 0xB5;  // micro
    codepoints[n++] = 0xB2;  // squared
    codepoints[n++] = 0xB1;  // plus-minus
    codepoints[n++] = 0xD7;  // times
    codepoints[n++] = 0x2212;  // minus
    const char* faces[2] = {"font.ttf", "font_bold.ttf"};
    Font* fonts[2] = {&V.font, &V.bold};
    for (int k = 0; k < 2; k++) {
        snprintf(path, sizeof(path), "%s/%s", folder, faces[k]);
        if (FileExists(path)) {
            *fonts[k] = LoadFontEx(path, 48, codepoints, n);
            SetTextureFilter(fonts[k]->texture, TEXTURE_FILTER_BILINEAR);
        } else {
            *fonts[k] = GetFontDefault();
        }
    }
    for (int k = 0; k < MAX_CALLOUTS; k++) {
        V.callout[k] = LoadRenderTexture(CALLOUT_W, CALLOUT_H);
        SetTextureFilter(V.callout[k].texture, TEXTURE_FILTER_BILINEAR);
    }
    V.speed = 1.0f;
    V.playing = 1;
    V.autoplay = 1;
    V.overlays = OV_FORCES | OV_SENSING | OV_HUD;
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
    int width = 1600, height = 900, select = 0, camera = 1, overlays = OV_FORCES | OV_SENSING | OV_HUD;
    float at = -1, zoom = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) folder = argv[++i];
        else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc) shot = argv[++i];
        else if (!strcmp(argv[i], "--episode") && i + 1 < argc) select = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i + 1 < argc) at = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--camera") && i + 1 < argc) camera = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--overlays") && i + 1 < argc) overlays = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--zoom") && i + 1 < argc) zoom = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 2 < argc) { width = atoi(argv[++i]); height = atoi(argv[++i]); }
    }
    init(folder, width, height);
    V.overlays = overlays;
    ni_select(select);
    if (at >= 0) { ni_seek(at); ni_play(0); }
    sample(V.time);
    camera_preset(camera);
    V.distance *= zoom;
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
