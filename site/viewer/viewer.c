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

#define MAX_MOVING 96
#define SHADOW_SIZE 4096
#define NJ 5
// Disturbance overlays are drawn at screen scale so they read at any zoom; the legend states the scales.
#define ARROW_PX_PER_N 7000.0f     // force arrows: 1 mN is 7 screen pixels
#define ARROW_PX_PER_ACC 15000.0f  // table vibration: 1 mm/s^2 is 15 screen pixels
#define OFFSET_PX_PER_10UM 80.0f   // micrometre offsets: magnified so 10 um spans about 80 pixels
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
    float* data;    // frames x stride, see the X_ (alignment) or E_ (insertion) offsets
    // Insertion runs: placed threads, and event times and places for the effects.
    unsigned nplaced;
    int site[3];
    float placeDepth[3], placeLateral[3];
    int nevent;
    float eventTime[16];
    int eventKind[16];  // 0 bond formed, 1 puncture, 2 release
    Vector3 eventAt[16];
} Episode;

#define MAX_BODIES 1024

// One dataset: a scene and its recorded runs. The viewer holds two (insertion and alignment) and
// swaps them when the page changes mode.
typedef struct {
    int kind;  // KIND_ALIGN or KIND_TUBE
    Body* bodies;
    int nbody;
    Mat* mats;
    int nmat;
    Chunk* chunks;
    int nchunk;
    Episode* episodes;
    int nepisode, nmoving, stride, extra, selected;
    int moving[MAX_MOVING];
    Vector3 axis[MAX_MOVING], centroid[MAX_MOVING];
    Vector3 current[MAX_MOVING];
    Quaternion currentQ[MAX_MOVING];
    int threadOf[MAX_MOVING];   // insertion: thread index of a moving body, -1 otherwise
    int movingOf[MAX_BODIES];   // body -> moving index, -1 when fixed
    int glass, tissue;          // material indices drawn translucent (insertion), -1 if absent
    float segment;              // thread segment length, m
    int loaded;
} Set;

enum { KIND_ALIGN = 0, KIND_TUBE = 1 };

// Frame layout after time, moving-body positions and tip (site_export.EXTRA).
enum { X_LAT, X_VERT, X_GOAL = 2, X_MTIP = 5, X_MGOAL = 8, X_ACC = 11, X_FORCE = 14, X_FRICTION = 19,
       X_INERTIAL = 24, X_ACTION = 29, X_COUNT = 30 };

// Insertion frame layout after time and the moving-body poses (site_export.TUBE_EXTRA).
enum { E_TIP = 0, E_END = 3, E_PHASE = 6, E_BOND = 7, E_THREAD = 8, E_DEPTH = 9, E_AXIAL = 10, E_PUNCT = 11,
       E_TISSUE = 12, E_TARGET = 15, E_MTARGET = 18, E_MTIP = 21, E_ACC = 24, E_NOISE = 27, E_FRICTION = 31,
       E_INERTIAL = 35, E_PLACE = 39, E_SITE = 40, E_COUNT = 41 };
#define TUBE_SLIDES 4
static const char* PHASE_NAMES[11] = {"ready", "reload", "move", "descend", "correct", "needle down", "insert",
                                      "release", "snap back", "lift", "learned approach"};
#define PHASE_LEARNED 10
static const char* CONTROLLER_NAMES[2] = {"LEARNED POLICY insert_v2", "SCRIPTED YARDSTICK"};
static int needle_phase(int phase) { return phase >= 5 && phase <= 8; }

static struct {
    Set D, alt;  // the shown dataset and the other one
    int mode;    // 0 insertion, 1 alignment
    float time, speed, hold;
    int playing, autoplay;
    Shader pbr, depth;
    Material material, depthMaterial;
    RenderTexture2D shadow;
    int loc[16];
    Matrix lightVP;
    Camera3D camera;
    float yaw, pitch, distance;
    float zoomTo;  // orbit distance the camera eases toward (zoom input moves this, clamped)
    Vector3 target;
    int follow;
    Vector3 tip;
    float lateral, vertical;
    // Disturbance overlays.
    int overlays, camPreset;
    float x[64];  // interpolated extra fields at the display time (X_ or E_ offsets)
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
    int wide;        // has text values: the value column is sized for the longest text value
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

enum { L_ALBEDO, L_SURFACE, L_CAMPOS, L_LIGHTDIR, L_LIGHTCOLOR, L_FILLDIR, L_FILLCOLOR, L_EXPOSURE, L_LIGHTVP, L_TEXEL, L_BIAS, L_OPACITY };

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
    V.D.nbody = (int)u32(&r);
    V.D.nmat = (int)u32(&r);
    V.D.nchunk = (int)u32(&r);
    V.D.bodies = calloc(V.D.nbody, sizeof(Body));
    for (int b = 0; b < V.D.nbody; b++) {
        i32(&r);
        take(&r, V.D.bodies[b].name, 32);
        V.D.bodies[b].pos = (Vector3){f32(&r), f32(&r), f32(&r)};
        float w = f32(&r), x = f32(&r), y = f32(&r), z = f32(&r);  // MuJoCo w, x, y, z
        V.D.bodies[b].quat = (Quaternion){x, y, z, w};
        unsigned char moving;
        take(&r, &moving, 1);
        V.D.bodies[b].moving = moving;
    }
    V.D.mats = calloc(V.D.nmat, sizeof(Mat));
    for (int m = 0; m < V.D.nmat; m++) {
        take(&r, V.D.mats[m].name, 16);
        for (int k = 0; k < 3; k++) V.D.mats[m].albedo[k] = f32(&r);
        for (int k = 0; k < 4; k++) V.D.mats[m].surface[k] = f32(&r);
    }
    V.D.chunks = calloc(V.D.nchunk, sizeof(Chunk));
    for (int c = 0; c < V.D.nchunk; c++) {
        Chunk* ch = &V.D.chunks[c];
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
    V.D.glass = V.D.tissue = -1;
    for (int m = 0; m < V.D.nmat; m++) {
        if (!strcmp(V.D.mats[m].name, "glass")) V.D.glass = m;
        if (!strcmp(V.D.mats[m].name, "tissue")) V.D.tissue = m;
    }
}

static void map_moving(void) {
    for (int b = 0; b < MAX_BODIES; b++) V.D.movingOf[b] = -1;
    for (int m = 0; m < V.D.nmoving; m++) {
        if (V.D.moving[m] >= 0 && V.D.moving[m] < MAX_BODIES) V.D.movingOf[V.D.moving[m]] = m;
        V.D.threadOf[m] = -1;
    }
}

static void moving_centroids(void) {
    // Arrow anchors: centroid of each moving body's geometry, in body coordinates.
    for (int m = 0; m < V.D.nmoving; m++) {
        double sum[3] = {0, 0, 0};
        long n = 0;
        for (int c = 0; c < V.D.nchunk; c++) {
            if (V.D.chunks[c].body != V.D.moving[m]) continue;
            Mesh* mesh = &V.D.chunks[c].mesh;
            for (int v = 0; v < mesh->vertexCount; v++, n++) {
                for (int k = 0; k < 3; k++) sum[k] += mesh->vertices[3 * v + k];
            }
        }
        V.D.centroid[m] = n ? (Vector3){(float)(sum[0] / n), (float)(sum[1] / n), (float)(sum[2] / n)} : Vector3Zero();
    }
}

static void load_replays(const char* path) {
    Reader r = open_file(path, "NIR2");
    V.D.nepisode = (int)u32(&r);
    V.D.nmoving = (int)u32(&r);
    V.D.extra = (int)u32(&r);
    if (V.D.nmoving > MAX_MOVING || V.D.extra != X_COUNT) {
        fprintf(stderr, "viewer: unexpected replay layout\n");
        exit(2);
    }
    for (int k = 0; k < V.D.nmoving; k++) V.D.moving[k] = i32(&r);
    for (int k = 0; k < V.D.nmoving; k++) V.D.axis[k] = (Vector3){f32(&r), f32(&r), f32(&r)};
    V.D.stride = 1 + 3 * V.D.nmoving + 3 + V.D.extra;
    V.D.episodes = calloc(V.D.nepisode, sizeof(Episode));
    int o = 1 + 3 * V.D.nmoving + 3;
    for (int e = 0; e < V.D.nepisode; e++) {
        Episode* ep = &V.D.episodes[e];
        ep->seed = u32(&r);
        ep->target = u32(&r);
        ep->policy = u32(&r);
        ep->outcome = u32(&r);
        ep->frames = u32(&r);
        ep->level = f32(&r);
        ep->latency = f32(&r);
        ep->data = malloc(sizeof(float) * V.D.stride * ep->frames);
        take(&r, ep->data, sizeof(float) * V.D.stride * ep->frames);
        for (int k = 0; k < 3; k++) {
            double sum = 0;
            for (unsigned f = 0; f < ep->frames; f++) sum += ep->data[f * V.D.stride + o + X_GOAL + k];
            ep->goal[k] = (float)(sum / ep->frames);
        }
    }
    UnloadFileData(r.data);
    V.D.kind = KIND_ALIGN;
    map_moving();
    moving_centroids();
    V.D.loaded = 1;
}

// Insertion runs (NIT2): poses with quaternions for every moving body, threads included, and the
// controller of each run (0 learned policy, 1 scripted yardstick).
static void load_tube_replays(const char* path) {
    Reader r = open_file(path, "NIT2");
    V.D.nepisode = (int)u32(&r);
    V.D.nmoving = (int)u32(&r);
    V.D.extra = (int)u32(&r);
    if (V.D.nmoving > MAX_MOVING || V.D.extra != E_COUNT) {
        fprintf(stderr, "viewer: unexpected insertion replay layout\n");
        exit(2);
    }
    for (int k = 0; k < V.D.nmoving; k++) V.D.moving[k] = i32(&r);
    for (int k = 0; k < TUBE_SLIDES; k++) V.D.axis[k] = (Vector3){f32(&r), f32(&r), f32(&r)};
    V.D.stride = 1 + 7 * V.D.nmoving + V.D.extra;
    V.D.episodes = calloc(V.D.nepisode, sizeof(Episode));
    int o = 1 + 7 * V.D.nmoving;
    for (int e = 0; e < V.D.nepisode; e++) {
        Episode* ep = &V.D.episodes[e];
        ep->seed = u32(&r);
        ep->policy = u32(&r);
        ep->level = f32(&r);
        ep->latency = f32(&r);
        f32(&r);  // simulated seconds
        ep->outcome = u32(&r);
        ep->nplaced = u32(&r);
        for (int k = 0; k < 3; k++) {
            ep->site[k] = i32(&r);
            ep->placeDepth[k] = f32(&r);
            ep->placeLateral[k] = f32(&r);
        }
        ep->frames = u32(&r);
        ep->data = malloc(sizeof(float) * V.D.stride * ep->frames);
        take(&r, ep->data, sizeof(float) * V.D.stride * ep->frames);
        ep->target = ep->site[0] >= 0 ? (unsigned)ep->site[0] : 0;
        for (int k = 0; k < 3; k++) ep->goal[k] = ep->data[o + E_TARGET + k];
        // Events for the effects: bond formed, puncture, release.
        ep->nevent = 0;
        for (unsigned f = 1; f < ep->frames && ep->nevent < 16; f++) {
            const float* p = ep->data + (f - 1) * V.D.stride + o;
            const float* q = ep->data + f * V.D.stride + o;
            int kind = -1;
            Vector3 at = {q[E_END], q[E_END + 1], q[E_END + 2]};
            if (p[E_BOND] < 0.5f && q[E_BOND] > 0.5f && q[E_BOND] < 1.5f) kind = 0;
            else if (p[E_PUNCT] < 0.5f && q[E_PUNCT] > 0.5f) { kind = 1; at = (Vector3){q[E_TIP], q[E_TIP + 1], q[E_TIP + 2]}; }
            else if (p[E_BOND] < 1.5f && q[E_BOND] > 1.5f) kind = 2;
            if (kind < 0) continue;
            ep->eventTime[ep->nevent] = ep->data[f * V.D.stride];
            ep->eventKind[ep->nevent] = kind;
            ep->eventAt[ep->nevent++] = at;
        }
    }
    UnloadFileData(r.data);
    V.D.kind = KIND_TUBE;
    map_moving();
    for (int m = 0; m < V.D.nmoving; m++) {
        const char* name = V.D.bodies[V.D.moving[m]].name;
        if (!strncmp(name, "thread_B", 8)) {
            const char* u = strrchr(name, '_');
            V.D.threadOf[m] = u ? atoi(u + 1) : -1;
        }
    }
    moving_centroids();
    // Segment length from the first two bodies of thread 0 in the first recorded frame.
    V.D.segment = 4.6e-4f;
    for (int m = 0; m + 1 < V.D.nmoving; m++) {
        if (V.D.threadOf[m] == 0 && V.D.threadOf[m + 1] == 0) {
            const float* p = V.D.episodes[0].data + 1 + 7 * m;
            V.D.segment = Vector3Distance((Vector3){p[0], p[1], p[2]}, (Vector3){p[7], p[8], p[9]});
            break;
        }
    }
    V.D.loaded = 1;
}

// ---------------------------------------------------------------- playback

EXPORT void ni_select(int i);

static Episode* episode(void) { return &V.D.episodes[V.D.selected]; }

static float duration(void) {
    Episode* e = episode();
    return e->data[(e->frames - 1) * V.D.stride];
}

static void sample_tube(float t) {
    Episode* e = episode();
    int k = 0;
    while (k + 1 < (int)e->frames && e->data[(k + 1) * V.D.stride] <= t) k++;
    int j = k + 1 < (int)e->frames ? k + 1 : k;
    float t0 = e->data[k * V.D.stride], t1 = e->data[j * V.D.stride];
    float a = (j > k && t1 > t0) ? Clamp((t - t0) / (t1 - t0), 0, 1) : 0;
    const float* p = e->data + k * V.D.stride;
    const float* q = e->data + j * V.D.stride;
    for (int m = 0; m < V.D.nmoving; m++) {
        const float* u = p + 1 + 7 * m;
        const float* w = q + 1 + 7 * m;
        V.D.current[m] = Vector3Lerp((Vector3){u[0], u[1], u[2]}, (Vector3){w[0], w[1], w[2]}, a);
        Quaternion qa = {u[4], u[5], u[6], u[3]}, qb = {w[4], w[5], w[6], w[3]};  // MuJoCo w, x, y, z
        V.D.currentQ[m] = QuaternionSlerp(qa, qb, a);
    }
    int o = 1 + 7 * V.D.nmoving;
    const float* n = a < 0.5f ? p : q;
    for (int x = 0; x < E_COUNT; x++) {
        int discrete = x == E_PHASE || x == E_BOND || x == E_THREAD || x == E_PUNCT || x == E_SITE;
        float u = p[o + x], w = q[o + x];
        V.x[x] = discrete || isnan(u) || isnan(w) ? n[o + x] : u + a * (w - u);
    }
    V.tip = (Vector3){V.x[E_TIP], V.x[E_TIP + 1], V.x[E_TIP + 2]};
    V.lateral = n[o + E_PLACE] * 1e6f;
    V.vertical = n[o + E_DEPTH] * 1e6f;
}

static void sample(float t) {
    if (V.D.kind == KIND_TUBE) { sample_tube(t); return; }
    Episode* e = episode();
    int k = 0;
    while (k + 1 < (int)e->frames && e->data[(k + 1) * V.D.stride] <= t) k++;
    int j = k + 1 < (int)e->frames ? k + 1 : k;
    float t0 = e->data[k * V.D.stride], t1 = e->data[j * V.D.stride];
    float a = (j > k && t1 > t0) ? Clamp((t - t0) / (t1 - t0), 0, 1) : 0;
    const float* p = e->data + k * V.D.stride;
    const float* q = e->data + j * V.D.stride;
    for (int m = 0; m < V.D.nmoving; m++) {
        V.D.current[m] = Vector3Lerp((Vector3){p[1 + 3 * m], p[2 + 3 * m], p[3 + 3 * m]},
                                   (Vector3){q[1 + 3 * m], q[2 + 3 * m], q[3 + 3 * m]}, a);
    }
    int o = 1 + 3 * V.D.nmoving;
    V.tip = Vector3Lerp((Vector3){p[o], p[o + 1], p[o + 2]}, (Vector3){q[o], q[o + 1], q[o + 2]}, a);
    for (int k = 0; k < X_COUNT; k++) V.x[k] = p[o + 3 + k] + a * (q[o + 3 + k] - p[o + 3 + k]);
    // Error readouts come from the nearest recorded state, not the interpolation.
    const float* n = a < 0.5f ? p : q;
    V.lateral = n[o + 3 + X_LAT];
    V.vertical = n[o + 3 + X_VERT];
}

static Vector3 xvec(int k) { return (Vector3){V.x[k], V.x[k + 1], V.x[k + 2]}; }

static Vector3 frame_vec(const Episode* e, int f, int k) {
    const float* p = e->data + f * V.D.stride + 1 + 3 * V.D.nmoving + 3 + k;
    return (Vector3){p[0], p[1], p[2]};
}

static Matrix body_transform(int b) {
    Vector3 pos = V.D.bodies[b].pos;
    Quaternion quat = V.D.bodies[b].quat;
    int m = b < MAX_BODIES ? V.D.movingOf[b] : -1;
    if (m >= 0) {
        pos = V.D.current[m];
        if (V.D.kind == KIND_TUBE) quat = V.D.currentQ[m];
    }
    return MatrixMultiply(QuaternionToMatrix(quat), MatrixTranslate(pos.x, pos.y, pos.z));
}

// Insertion: a spare thread is hidden until it is reloaded into the tube (the parked spares are a stand-in).
static int hidden(int body) {
    if (V.D.kind != KIND_TUBE || body >= MAX_BODIES) return 0;
    int m = V.D.movingOf[body];
    return m >= 0 && V.D.threadOf[m] > (int)(V.x[E_THREAD] + 0.5f);
}

// ---------------------------------------------------------------- camera

// Zoom: input changes a target orbit distance in log space; the camera eases toward it each frame.
// Steps toward a limit shrink as the target nears it (a soft end, no overshoot), and the target
// itself is clamped, so nothing accumulates past a limit and reversing responds at once.
#define ZOOM_MIN 0.0003f      // m: 0.3 mm from the target
#define ZOOM_MAX 3.0f
#define ZOOM_MARGIN 0.7f      // log-distance range over which the ends resist
#define ZOOM_EASE_S 0.07f     // time constant of the camera following the target

static void zoom_by(float du) {
    const float lo = logf(ZOOM_MIN), hi = logf(ZOOM_MAX);
    float u = logf(V.zoomTo);
    float room = du > 0 ? hi - u : u - lo;
    du *= Clamp(room / ZOOM_MARGIN, 0.0f, 1.0f);
    V.zoomTo = expf(Clamp(u + du, lo, hi));
}

static void camera_preset_set(int preset);
static void camera_preset(int preset) {
    camera_preset_set(preset);
    V.zoomTo = V.distance;
}

static void camera_preset_set(int preset) {
    V.follow = preset == 2;
    V.camPreset = preset;
    if (V.D.kind == KIND_TUBE && preset > 0) {
        if (preset == 1) {
            V.target = (Vector3){0.0f, -0.052f, 0.113f}; V.distance = 0.075f; V.yaw = -2.10f; V.pitch = 0.52f;
        } else if (preset == 2) {
            V.target = V.tip; V.distance = 0.014f; V.yaw = -2.20f; V.pitch = 0.22f;
        } else {
            V.follow = 3;  // follow the point between the needle and the thread end, looking down through the tissue
            V.target = V.tip; V.distance = 0.0045f; V.yaw = -2.35f; V.pitch = 0.62f;
        }
        return;
    }
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
        // Grab-style pan: the scene follows the cursor on both axes.
        V.target = Vector3Add(V.target, Vector3Add(Vector3Scale(right, -delta.x * s), Vector3Scale(up, delta.y * s)));
        V.follow = 0;
    }
#ifndef PLATFORM_WEB
    float wheel = GetMouseWheelMove();  // the web page sends wheel and pinch input through ni_zoom instead
    if (wheel != 0) zoom_by(-wheel * 0.12f);
#endif
    if (V.zoomTo <= 0) V.zoomTo = V.distance;
    float ease = 1.0f - expf(-fminf(GetFrameTime(), 0.1f) / ZOOM_EASE_S);
    V.distance = expf(logf(V.distance) + (logf(V.zoomTo) - logf(V.distance)) * ease);
    if (V.follow == 3) {
        Vector3 end = {V.x[E_END], V.x[E_END + 1], V.x[E_END + 2]};
        V.target = Vector3Lerp(V.tip, end, 0.5f);
    } else if (V.follow) {
        V.target = V.tip;
    }
    V.camera.target = V.target;
    V.camera.position = Vector3Add(V.target, Vector3Scale(forward, V.distance));
    V.camera.up = (Vector3){0, 0, 1};
    V.camera.fovy = 40.0f;
    V.camera.projection = CAMERA_PERSPECTIVE;
}

// ---------------------------------------------------------------- drawing

static void set_vec3(int loc, Vector3 v) { SetShaderValue(V.pbr, V.loc[loc], &v, SHADER_UNIFORM_VEC3); }

// Glass is always translucent; tissue too in the insertion Micro view, to show the thread inside it.
static float chunk_opacity(const Chunk* ch) {
    if (V.D.kind != KIND_TUBE) return 1.0f;
    if (ch->mat == V.D.glass) return 0.30f;
    if (ch->mat == V.D.tissue && V.camPreset == 3) return 0.42f;
    return 1.0f;
}

static void draw_chunks_pass(Material material, int shade, int translucent) {
    for (int c = 0; c < V.D.nchunk; c++) {
        Chunk* ch = &V.D.chunks[c];
        if (hidden(ch->body)) continue;
        float opacity = chunk_opacity(ch);
        if (shade && (opacity < 1.0f) != translucent) continue;
        if (shade) {
            SetShaderValue(V.pbr, V.loc[L_OPACITY], &opacity, SHADER_UNIFORM_FLOAT);
            Mat* m = &V.D.mats[ch->mat];
            SetShaderValue(V.pbr, V.loc[L_ALBEDO], m->albedo, SHADER_UNIFORM_VEC3);
            SetShaderValue(V.pbr, V.loc[L_SURFACE], m->surface, SHADER_UNIFORM_VEC4);
        }
        DrawMesh(ch->mesh, material, body_transform(ch->body));
    }
}

static void draw_chunks(Material material, int shade) {
    draw_chunks_pass(material, shade, 0);
    if (!shade) return;
    rlDrawRenderBatchActive();
    BeginBlendMode(BLEND_ALPHA);
    rlDisableDepthMask();
    draw_chunks_pass(material, shade, 1);
    rlDrawRenderBatchActive();
    rlEnableDepthMask();
    EndBlendMode();
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
    int o = 1 + 3 * V.D.nmoving;
    Color trail = (Color){211, 138, 170, 255};
    for (int k = 0; k + 1 < (int)e->frames && e->data[(k + 1) * V.D.stride] <= V.time; k++) {
        const float* p = e->data + k * V.D.stride;
        const float* q = e->data + (k + 1) * V.D.stride;
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

// Magnification for micrometre offsets (sensing errors, tissue motion) at the current zoom: 10 um spans
// about OFFSET_PX_PER_10UM pixels, rounded to the nearest 1, 2 or 5 times a power of ten (on a log
// scale), never below 1.
static float magnify(void) {
    float raw = OFFSET_PX_PER_10UM * px() / 10e-6f;
    if (raw <= 1) return 1;
    float p = powf(10.0f, floorf(log10f(raw))), best = p, err = 1e9f;
    const float steps[4] = {1, 2, 5, 10};
    for (int k = 0; k < 4; k++) {
        float e = fabsf(logf(raw / (p * steps[k])));
        if (e < err) { err = e; best = p * steps[k]; }
    }
    return best;
}

// A point drawn `m` times farther from its reference than it is.
static Vector3 mag(Vector3 ref, Vector3 p, float m) { return Vector3Add(ref, Vector3Scale(Vector3Subtract(p, ref), m)); }
static float force_scale(void) { return ARROW_PX_PER_N * px(); }
static float acc_scale(void) { return ARROW_PX_PER_ACC * px(); }

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

// A dashed line with thickness (radius in world units), for the offsets the overlays exist to show.
static void dashed_thick(Vector3 a, Vector3 b, float dash, float radius, Color c) {
    float len = Vector3Distance(a, b);
    int n = (int)fminf(len / dash, 200);
    if (n < 2) { DrawCylinderEx(a, b, radius, radius, 6, c); return; }
    for (int k = 0; k < n; k += 2) {
        DrawCylinderEx(Vector3Lerp(a, b, (float)k / n), Vector3Lerp(a, b, fminf((float)(k + 1) / n, 1)), radius, radius, 6, c);
    }
}

static Color fade(Color c, float a) { c.a = (unsigned char)(255 * Clamp(a, 0, 1)); return c; }

static Vector3 vibration_anchor(void) { return (Vector3){0.112f, -0.040f, 0.004f}; }

static void draw_forces(void) {
    float r = px() * 2.2f, fs = force_scale(), as = acc_scale();
    for (int m = 0; m < V.D.nmoving && m < NJ; m++) {
        Vector3 anchor = Vector3Transform(V.D.centroid[m], body_transform(V.D.moving[m]));
        Vector3 ax = V.D.axis[m];
        Vector3 side = Vector3Normalize(Vector3CrossProduct(ax, fabsf(ax.z) > 0.9f ? (Vector3){1, 0, 0} : (Vector3){0, 0, 1}));
        float comps[3] = {V.x[X_FORCE + m], V.x[X_FRICTION + m], V.x[X_INERTIAL + m]};
        Color colors[3] = {C_NOISE, C_FRICTION, C_VIBRATION};
        for (int k = 0; k < 3; k++) {
            Vector3 from = Vector3Add(anchor, Vector3Scale(side, (k - 1) * r * 4.0f));
            arrow(from, Vector3Scale(ax, comps[k] * fs), r, colors[k]);
        }
    }
    // Table vibration: acceleration arrow and the last 0.5 s of recorded samples.
    Episode* e = episode();
    Vector3 base = vibration_anchor();
    arrow(base, Vector3Scale(xvec(X_ACC), as), r * 1.2f, C_VIBRATION);
    int now = 0;
    while (now + 1 < (int)e->frames && e->data[(now + 1) * V.D.stride] <= V.time) now++;
    for (int f = (now > 15 ? now - 15 : 0); f < now; f++) {
        Vector3 a = Vector3Add(base, Vector3Scale(frame_vec(e, f, X_ACC), as));
        Vector3 b = Vector3Add(base, Vector3Scale(frame_vec(e, f + 1, X_ACC), as));
        DrawLine3D(a, b, fade(C_VIBRATION, 0.1f + 0.5f * (f - now + 15) / 15.0f));
    }
    DrawSphere(base, r * 1.5f, C_VIBRATION);
}

static void draw_sensing(void) {
    Episode* e = episode();
    int now = 0;
    while (now + 1 < (int)e->frames && e->data[(now + 1) * V.D.stride] <= V.time) now++;
    float s = px() * 9.0f, m = magnify();
    Vector3 goal = xvec(X_GOAL), estimate = mag(goal, xvec(X_MGOAL), m), measured = mag(V.tip, xvec(X_MTIP), m);
    // True target path over the last 2 s (breathing and pulse), magnified about the current target.
    for (int f = (now > 100 ? now - 100 : 0); f < now; f++) {
        DrawLine3D(mag(goal, frame_vec(e, f, X_GOAL), m), mag(goal, frame_vec(e, f + 1, X_GOAL), m),
                   fade(C_TRUTH, 0.15f + 0.6f * (f - now + 100) / 100.0f));
    }
    // Recent target estimates as a fading cloud (noise, bias and drift), each error magnified from its target.
    for (int f = (now > 25 ? now - 25 : 0); f <= now; f++) {
        Vector3 err = Vector3Subtract(frame_vec(e, f, X_MGOAL), frame_vec(e, f, X_GOAL));
        cross(Vector3Add(goal, Vector3Scale(err, m)), s * 0.35f, fade(C_ESTIMATE, 0.2f + 0.6f * (f - now + 25) / 25.0f));
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
    dashed_thick(goal, estimate, s * 0.4f, px() * 1.0f, fade(C_ESTIMATE, 0.9f));
    // What the policy is told about its tip: delayed by the latency and noisy.
    DrawSphereWires(measured, s * 0.6f, 6, 10, C_MEASURED);
    dashed_thick(V.tip, measured, s * 0.4f, px() * 1.0f, fade(C_MEASURED, 0.9f));
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
    if (value && value[0] && !strchr("0123456789+-\xe2\xc2", value[0])) c->wide = 1;  // a word, not a number
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
#define VALUE_TEMPLATE "\xe2\x88\x92" "000.0 \xc2\xb5m"  // widest number any callout shows
#define WORD_TEMPLATE "learned approach"                  // widest word any callout shows

static void collect_callouts(void) {
    NC = 0;
    Episode* e = episode();
    float d = V.distance;
    char b[32];
    const char* names[NJ] = {"X slide", "Y slide", "Z slide", "insertion slide", "retainer slide"};
    const float off[NJ][2] = {{0.16f, 0.07f}, {-0.22f, 0.05f}, {0.17f, -0.02f}, {-0.20f, -0.07f}, {0.15f, -0.24f}};
    if ((V.overlays & OV_FORCES) && d > 0.5f) {
        for (int m = 0; m < V.D.nmoving && m < NJ; m++) {
            Vector3 anchor = Vector3Transform(V.D.centroid[m], body_transform(V.D.moving[m]));
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
    float ww = fmaxf(tw, MeasureTextEx(V.font, WORD_TEMPLATE, CALLOUT_FONT, 0).x);
    for (int k = 0; k < NC; k++) {
        Callout* c = &C[k];
        // Width from the title, the (constant) labels and the template value only.
        float w = MeasureTextEx(V.bold, c->label[0], CALLOUT_FONT, 0).x;
        for (int i = 1; i < c->lines; i++) w = fmaxf(w, MeasureTextEx(V.font, c->label[i], CALLOUT_FONT, 0).x + 24 + (c->wide ? ww : tw));
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
static const Color C_LEADER = {206, 200, 232, 255};  // callout leader lines and anchors

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
        // Leaders in one neutral colour, so no line in the scene can be mistaken for the lime thread;
        // the panel's stripe and title carry the callout's colour.
        DrawLine3D(c->anchor, pos, fade(C_LEADER, 0.85f));
        DrawSphere(c->anchor, scale * 0.0016f, C_LEADER);
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
    float pw = 300 * s, chart = 84 * s, ph = line * 9.4f + chart + 16 * s;
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
    snprintf(buf, sizeof(buf), "drawn larger: offsets \xc3\x97%g, arrows 1 mN = 7 px", magnify());
    DrawTextEx(V.font, buf, (Vector2){x, y}, size * 0.85f, 0, (Color){200, 196, 226, 255});
    y += line * 1.05f;
    DrawTextEx(V.font, "height vs mean target, \xc2\xb1" "40 \xc2\xb5m", (Vector2){x, y}, size * 0.85f, 0, (Color){200, 196, 226, 255});
    y += line;
    Rectangle box = {x, y, iw, chart};
    DrawRectangleRec(box, (Color){12, 11, 26, 150});
    float span = 40e-6f, total = e->data[(e->frames - 1) * V.D.stride];
    float mid = box.y + box.height * 0.5f;
    Vector3 mean = {e->goal[0], e->goal[1], e->goal[2]};
    DrawLine((int)box.x, (int)mid, (int)(box.x + box.width), (int)mid, (Color){90, 86, 130, 255});
    int fields[3] = {X_GOAL, X_MGOAL, X_MTIP};
    Color colors[4] = {C_TRUTH, C_ESTIMATE, C_MEASURED, C_TIP};
    for (int series = 0; series < 4; series++) {
        Vector2 prev = {0};
        for (unsigned f = 0; f < e->frames; f++) {
            float t = e->data[f * V.D.stride];
            if (t > V.time) break;
            float z = series < 3 ? frame_vec(e, f, fields[series]).z : e->data[f * V.D.stride + 1 + 3 * V.D.nmoving + 2];
            Vector2 p = {box.x + box.width * t / fmaxf(total, 1e-3f),
                         mid - box.height * 0.5f * Clamp((z - mean.z) / span, -1, 1)};
            if (f > 0) DrawLineEx(prev, p, series == 0 ? 2.0f * s : 1.3f * s, fade(colors[series], series == 1 ? 0.7f : 1));
            prev = p;
        }
    }
}


// ---------------------------------------------------------------- insertion runs

static const Color C_LIME = {92, 255, 20, 255};    // the thread
static const Color C_PUNCTURE = {255, 236, 246, 255};

static Vector3 ev(int k) { return (Vector3){V.x[k], V.x[k + 1], V.x[k + 2]}; }
static int has(int k) {
    if (isnan(V.x[k]) || isnan(V.x[k + 1]) || isnan(V.x[k + 2])) return 0;
    int phase = (int)(V.x[E_PHASE] + 0.5f);
    // The target estimate counts once it refers to the current site; the tip measurement only while the
    // robot measures (the yardstick's descend and correct; every step of the learned approach).
    if (k == E_MTARGET) return Vector3Distance(ev(E_MTARGET), ev(E_TARGET)) < 1e-3f;
    if (k == E_MTIP) return phase == 3 || phase == 4 || phase == PHASE_LEARNED;
    return 1;
}

static Vector3 tube_frame_vec(const Episode* e, int f, int k) {
    const float* p = e->data + f * V.D.stride + 1 + 7 * V.D.nmoving + k;
    return (Vector3){p[0], p[1], p[2]};
}

static int tube_now(const Episode* e) {
    int now = 0;
    while (now + 1 < (int)e->frames && e->data[(now + 1) * V.D.stride] <= V.time) now++;
    return now;
}

// Threads as lines through their segment origins: at workcell scale a 40 µm thread is thinner than a
// pixel, so the line keeps it visible. The geometry itself is drawn at true size.
static void draw_threads(void) {
    int shown = (int)(V.x[E_THREAD] + 0.5f);
    for (int t = 0; t <= shown; t++) {
        Vector3 prev = {0};
        int have = 0, last = -1;
        for (int m = 0; m < V.D.nmoving; m++) {
            if (V.D.threadOf[m] != t) continue;
            if (have) DrawLine3D(prev, V.D.current[m], C_LIME);
            prev = V.D.current[m];
            have = 1;
            last = m;
        }
        if (last < 0) continue;
        Vector3 end = t == shown ? ev(E_END)
                                 : Vector3Add(prev, Vector3RotateByQuaternion((Vector3){V.D.segment, 0, 0}, V.D.currentQ[last]));
        DrawLine3D(prev, end, C_LIME);
    }
}

static void draw_tube_trail(void) {
    Episode* e = episode();
    int now = tube_now(e);
    Color trail = (Color){211, 138, 170, 255};
    for (int f = (now > 400 ? now - 400 : 0); f < now; f++) {
        DrawLine3D(tube_frame_vec(e, f, E_TIP), tube_frame_vec(e, f + 1, E_TIP), fade(trail, 0.25f + 0.75f * (f - now + 400) / 400.0f));
    }
    draw_threads();
}

// Effects at recorded events: a flash where the needle and the thread stick, ripples on the tissue at
// puncture, a ring where the thread is released. Durations in simulated time.
static void draw_tube_effects(void) {
    Episode* e = episode();
    for (int k = 0; k < e->nevent; k++) {
        float age = V.time - e->eventTime[k];
        float life = e->eventKind[k] == 1 ? 0.030f : e->eventKind[k] == 0 ? 0.014f : 0.020f;
        if (age < 0 || age > life) continue;
        float u = age / life;
        Vector3 at = e->eventAt[k];
        if (e->eventKind[k] == 0) {
            DrawSphere(at, 30e-6f + 140e-6f * u, fade(C_LIME, 0.75f * (1 - u)));
            DrawSphereWires(at, 60e-6f + 260e-6f * u, 6, 12, fade(C_LIME, 1 - u));
        } else if (e->eventKind[k] == 1) {
            for (int r = 0; r < 3; r++) {
                float v = u - r * 0.18f;
                if (v <= 0) continue;
                DrawCircle3D(Vector3Add(at, (Vector3){0, 0, 15e-6f}), 1.3e-3f * v, (Vector3){1, 0, 0}, 0, fade(C_PUNCTURE, 0.9f * (1 - v)));
            }
        } else {
            DrawCircle3D(at, 60e-6f + 400e-6f * u, (Vector3){1, 0, 0}, 0, fade(C_LIME, 1 - u));
            DrawCircle3D(at, 40e-6f + 250e-6f * u, (Vector3){1, 0, 0}, 0, fade(WHITE, 0.8f * (1 - u)));
        }
    }
}

static Vector3 tube_vibration_anchor(void) { return (Vector3){0.112f, -0.040f, 0.004f}; }

static void draw_tube_forces(void) {
    float r = px() * 2.2f, fs = force_scale(), as = acc_scale();
    for (int m = 0; m < TUBE_SLIDES; m++) {
        Vector3 anchor = Vector3Transform(V.D.centroid[m], body_transform(V.D.moving[m]));
        Vector3 ax = V.D.axis[m];
        Vector3 side = Vector3Normalize(Vector3CrossProduct(ax, fabsf(ax.z) > 0.9f ? (Vector3){1, 0, 0} : (Vector3){0, 0, 1}));
        float comps[3] = {V.x[E_NOISE + m], V.x[E_FRICTION + m], V.x[E_INERTIAL + m]};
        Color colors[3] = {C_NOISE, C_FRICTION, C_VIBRATION};
        for (int k = 0; k < 3; k++) {
            Vector3 from = Vector3Add(anchor, Vector3Scale(side, (k - 1) * r * 4.0f));
            arrow(from, Vector3Scale(ax, comps[k] * fs), r, colors[k]);
        }
    }
    Episode* e = episode();
    Vector3 base = tube_vibration_anchor();
    arrow(base, Vector3Scale(ev(E_ACC), as), r * 1.2f, C_VIBRATION);
    int now = tube_now(e);
    for (int f = (now > 25 ? now - 25 : 0); f < now; f++) {
        Vector3 a = Vector3Add(base, Vector3Scale(tube_frame_vec(e, f, E_ACC), as));
        Vector3 b = Vector3Add(base, Vector3Scale(tube_frame_vec(e, f + 1, E_ACC), as));
        DrawLine3D(a, b, fade(C_VIBRATION, 0.1f + 0.5f * (f - now + 25) / 25.0f));
    }
    DrawSphere(base, r * 1.5f, C_VIBRATION);
}

static void draw_tube_sensing(void) {
    Episode* e = episode();
    int now = tube_now(e);
    float s = px() * 9.0f, m = magnify();
    Vector3 goal = ev(E_TARGET);
    // True target path over the last second (breathing and pulse move the tissue), magnified about the
    // current target. Frames are 2 ms apart during the needle's work and 8 ms between.
    for (int f = (now > 500 ? now - 500 : 0); f < now; f++) {
        if ((int)(e->data[f * V.D.stride + 1 + 7 * V.D.nmoving + E_SITE]) != (int)V.x[E_SITE]) continue;
        if (V.time - e->data[f * V.D.stride] > 1.0f) continue;
        DrawLine3D(mag(goal, tube_frame_vec(e, f, E_TARGET), m), mag(goal, tube_frame_vec(e, f + 1, E_TARGET), m),
                   fade(C_TRUTH, 0.15f + 0.6f * (1.0f - (V.time - e->data[f * V.D.stride]))));
    }
    DrawSphere(goal, s * 0.45f, C_TRUTH);
    if (has(E_MTARGET)) {
        Vector3 estimate = mag(goal, ev(E_MTARGET), m);
        cross(estimate, s, C_ESTIMATE);
        DrawSphereWires(estimate, s * 0.6f, 6, 10, C_ESTIMATE);
        dashed_thick(goal, estimate, s * 0.4f, px() * 1.0f, fade(C_ESTIMATE, 0.9f));
    }
    if (has(E_MTIP)) {
        Vector3 measured = mag(V.tip, ev(E_MTIP), m);
        DrawSphereWires(measured, s * 0.6f, 6, 10, C_MEASURED);
        dashed_thick(V.tip, measured, s * 0.4f, px() * 1.0f, fade(C_MEASURED, 0.9f));
    }
    DrawSphere(V.tip, s * 0.45f, C_TIP);
}

static void tube_callouts(void) {
    Episode* e = episode();
    float d = V.distance;
    char b[32], n[32];
    int phase = (int)(V.x[E_PHASE] + 0.5f), bond = (int)(V.x[E_BOND] + 0.5f), site = (int)(V.x[E_SITE] + 0.5f);
    const char* names[TUBE_SLIDES] = {"X slide", "Y slide", "Z slide", "insertion slide"};
    const float off[TUBE_SLIDES][2] = {{0.16f, 0.07f}, {-0.22f, 0.05f}, {0.17f, -0.02f}, {-0.20f, -0.07f}};
    if ((V.overlays & OV_FORCES) && d > 0.5f) {
        for (int m = 0; m < TUBE_SLIDES; m++) {
            Vector3 anchor = Vector3Transform(V.D.centroid[m], body_transform(V.D.moving[m]));
            Callout* c = callout(anchor, off[m][0], off[m][1], names[m], WHITE);
            snprintf(b, sizeof(b), "%s mN", num(n, V.x[E_NOISE + m] * 1e3f, 1, 1));
            row(c, C_NOISE, "noise", b);
            snprintf(b, sizeof(b), "%s mN", num(n, V.x[E_FRICTION + m] * 1e3f, 1, 1));
            row(c, C_FRICTION, "friction", b);
            snprintf(b, sizeof(b), "%s mN", num(n, V.x[E_INERTIAL + m] * 1e3f, 1, 1));
            row(c, C_VIBRATION, "vibration", b);
        }
        Callout* c = callout(tube_vibration_anchor(), 0.10f, 0.05f, "table vibration", WHITE);
        snprintf(b, sizeof(b), "%s mm/s\xc2\xb2", num(n, Vector3Length(ev(E_ACC)) * 1e3f, 1, 0));
        row(c, C_VIBRATION, "acceleration", b);
    }
    if (d > 0.5f) {
        Callout* c = callout(V.tip, -0.18f, 0.16f, "tool", C_TIP);
        row(c, C_INK, "phase", PHASE_NAMES[phase]);
        snprintf(b, sizeof(b), "%d", site);
        row(c, C_INK, "site", b);
        return;
    }
    Vector3 goal = ev(E_TARGET), end = ev(E_END);
    float motion = Vector3Length(ev(E_TISSUE));
    // Placed threads at earlier sites: depth and placement from the run's results.
    int placed = 0;
    for (int k = 0; k < (int)e->nplaced; k++) {
        if (e->site[k] < 0 || e->site[k] == site) continue;
        int done = 0;
        for (int v = 0; v < e->nevent; v++) done += e->eventKind[v] == 2 && e->eventTime[v] <= V.time;
        if (k >= done) continue;
        // The thread's end lies about its depth below its target marking.
        Vector3 at = {0};
        int count = 0;
        for (int v = 0; v < e->nevent; v++) {
            if (e->eventKind[v] == 2 && count++ == k) at = e->eventAt[v];
        }
        Callout* c = callout(Vector3Add(at, (Vector3){0, 0, e->placeDepth[k]}), 0.20f, 0.10f + 0.12f * placed++, "placed thread", C_LIME);
        snprintf(b, sizeof(b), "site %d", e->site[k]);
        row(c, C_INK, "at", b);
        row(c, C_INK, "depth", length(n, e->placeDepth[k], 0));
        row(c, C_INK, "from target", length(n, e->placeLateral[k], 0));
    }
    Callout* c = callout(V.tip, -0.30f, 0.17f, "needle", C_TIP);
    row(c, C_INK, "phase", PHASE_NAMES[phase]);
    if (d < 0.05f) {
        snprintf(b, sizeof(b), "%s mN", num(n, -V.x[E_AXIAL] * 1e3f, 2, 0));
        row(c, C_INK, "tissue force", b);
        if (has(E_MTIP)) row(c, C_MEASURED, "measured, off by", length(n, Vector3Distance(ev(E_MTIP), V.tip), 0));
    }
    c = callout(end, 0.28f, 0.14f, "thread end", C_LIME);
    row(c, C_INK, "state", bond == 0 ? "waiting" : bond == 1 ? "on the needle" : "released");
    row(c, C_INK, "depth", length(b, V.x[E_DEPTH], 1));
    row(c, C_INK, "from target", length(n, V.x[E_PLACE], 0));
    c = callout(goal, 0.30f, -0.12f, "target", WHITE);
    row(c, C_INK, "tissue motion", length(b, motion, 0));
    if (has(E_MTARGET)) row(c, C_ESTIMATE, "estimate off by", length(n, Vector3Distance(ev(E_MTARGET), goal), 0));
}

static void draw_tube_hud(void) {
    Episode* e = episode();
    int w = GetScreenWidth(), h = GetScreenHeight();
    float s = fminf(h / 720.0f, w / 900.0f), size = 14.0f * s, line = size * 1.36f;
    float pw = 300 * s, chart = 84 * s, ph = line * 12.35f + chart + 16 * s;
    float x = 14 * s, y = h - ph - 14 * s;
    DrawRectangleRounded((Rectangle){x, y, pw, ph}, 0.06f, 8, (Color){24, 22, 44, 190});
    x += 12 * s;
    y += 10 * s;
    float iw = pw - 24 * s;
    char buf[64];
    DrawTextEx(V.bold, CONTROLLER_NAMES[e->policy ? 1 : 0], (Vector2){x, y}, size, 0,
               e->policy ? (Color){226, 222, 246, 255} : (Color){201, 255, 233, 255});
    y += line * 1.15f;
    snprintf(buf, sizeof(buf), "DISTURBANCE LEVEL %.2f", e->level);
    DrawTextEx(V.bold, buf, (Vector2){x, y}, size, 0, (Color){255, 214, 232, 255});
    y += line * 1.15f;
    swatch_row(x, y, iw, size, C_LIME, "thread", ""); y += line;
    swatch_row(x, y, iw, size, C_NOISE, "slide force noise", ""); y += line;
    swatch_row(x, y, iw, size, C_FRICTION, "extra friction", ""); y += line;
    swatch_row(x, y, iw, size, C_VIBRATION, "table vibration", ""); y += line;
    swatch_row(x, y, iw, size, C_MEASURED, "tip as last measured (late, noisy)", ""); y += line;
    swatch_row(x, y, iw, size, C_ESTIMATE, "target as last estimated", ""); y += line;
    swatch_row(x, y, iw, size, C_TRUTH, "true target (tissue motion)", ""); y += line;
    snprintf(buf, sizeof(buf), "drawn larger: offsets \xc3\x97%g, arrows 1 mN = 7 px", magnify());
    DrawTextEx(V.font, buf, (Vector2){x, y}, size * 0.85f, 0, (Color){200, 196, 226, 255});
    y += line;
    DrawTextEx(V.font, "chart: thread depth to 2.5 mm, needle force to 2.5 mN", (Vector2){x, y}, size * 0.85f, 0, (Color){200, 196, 226, 255});
    y += line;
    Rectangle box = {x, y, iw, chart};
    DrawRectangleRec(box, (Color){12, 11, 26, 150});
    float total = e->data[(e->frames - 1) * V.D.stride];
    int o = 1 + 7 * V.D.nmoving;
    for (int series = 0; series < 2; series++) {
        Vector2 prev = {0};
        for (unsigned f = 0; f < e->frames; f++) {
            float t = e->data[f * V.D.stride];
            if (t > V.time) break;
            float v = series == 0 ? e->data[f * V.D.stride + o + E_DEPTH] / 2.5e-3f : -e->data[f * V.D.stride + o + E_AXIAL] / 2.5e-3f;
            Vector2 q = {box.x + box.width * t / fmaxf(total, 1e-3f), box.y + box.height * (1 - Clamp(v, 0, 1))};
            if (f > 0) DrawLineEx(prev, q, series == 0 ? 2.0f * s : 1.3f * s, series == 0 ? C_LIME : C_TIP);
            prev = q;
        }
    }
}

static void draw_overlays_3d(void) {
    rlDrawRenderBatchActive();
    rlDisableDepthTest();
    rlDisableBackfaceCulling();
    if (V.D.kind == KIND_TUBE) {
        draw_tube_effects();
        if (V.overlays & OV_FORCES) draw_tube_forces();
        if (V.overlays & OV_SENSING) draw_tube_sensing();
    } else {
        if (V.overlays & OV_FORCES) draw_forces();
        if (V.overlays & OV_SENSING) draw_sensing();
        draw_ruler();
    }
    draw_callouts();
    rlDrawRenderBatchActive();
    rlEnableBackfaceCulling();
    rlEnableDepthTest();
}

static void draw_overlays_2d(void) {
    if (V.overlays & OV_HUD) {
        if (V.D.kind == KIND_TUBE) draw_tube_hud();
        else draw_hud();
    }
}

static void draw_frame(void) {
    shadow_pass();
    NC = 0;
    if (V.D.kind == KIND_TUBE) tube_callouts();
    else collect_callouts();
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
    if (V.D.kind == KIND_TUBE) draw_tube_trail();
    else draw_trail();
    draw_overlays_3d();
    EndMode3D();
    draw_overlays_2d();
    EndDrawing();
}

static void step(void) {
    float dt = GetFrameTime();
    if (dt > 0.1f) dt = 0.1f;
    if (V.playing) {
        float speed = V.speed;
        if (V.D.kind == KIND_TUBE && speed <= 0) speed = needle_phase((int)(V.x[E_PHASE] + 0.5f)) ? 0.02f : 0.2f;
        if (V.time < duration()) {
            V.time = fminf(V.time + dt * speed, duration());
        } else if ((V.hold += dt) > 1.6f) {
            V.hold = 0;
            V.time = 0;
            if (V.autoplay) {
                // Advance to the next episode of the same policy.
                int policy = (int)episode()->policy;
                for (int k = 1; k <= V.D.nepisode; k++) {
                    int c = (V.D.selected + k) % V.D.nepisode;
                    if ((V.D.kind == KIND_TUBE || (int)V.D.episodes[c].policy == policy) && V.D.episodes[c].level == episode()->level) {
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

EXPORT int ni_episode_count(void) { return V.D.nepisode; }
EXPORT unsigned ni_seed(int i) { return V.D.episodes[i].seed; }
EXPORT int ni_policy(int i) { return (int)V.D.episodes[i].policy; }
EXPORT int ni_target(int i) { return (int)V.D.episodes[i].target; }
EXPORT int ni_outcome(int i) { return (int)V.D.episodes[i].outcome; }
EXPORT int ni_selected(void) { return V.D.selected; }
EXPORT void ni_select(int i) {
    if (i >= 0 && i < V.D.nepisode) {
        V.D.selected = i; V.time = 0; V.hold = 0;
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
EXPORT void ni_zoom(float du) { zoom_by(du); }  // du: change of log orbit distance, positive zooms out
EXPORT float ni_distance(void) { return V.distance; }
EXPORT float ni_zoom_target(void) { return V.zoomTo; }
EXPORT int ni_camera_preset(void) { return V.camPreset; }
EXPORT float ni_lateral(void) { return V.lateral; }
EXPORT float ni_vertical(void) { return V.vertical; }
EXPORT void ni_resize(int w, int h) { SetWindowSize(w, h); }
EXPORT float ni_level(int i) { return V.D.episodes[i].level; }
EXPORT float ni_latency(int i) { return V.D.episodes[i].latency; }
EXPORT void ni_overlays(int mask) { V.overlays = mask; }
EXPORT int ni_mode_get(void) { return V.mode; }
EXPORT void ni_mode(int mode) {
    if (mode == V.mode || !V.alt.loaded) return;
    Set t = V.D; V.D = V.alt; V.alt = t;
    V.mode = mode;
    V.time = 0; V.hold = 0;
    sample(0);
    camera_preset(1);
}
EXPORT int ni_tube_phase(void) { return (int)(V.x[E_PHASE] + 0.5f); }
EXPORT int ni_tube_bond(void) { return (int)(V.x[E_BOND] + 0.5f); }
EXPORT int ni_tube_site(void) { return (int)(V.x[E_SITE] + 0.5f); }
EXPORT int ni_tube_thread(void) { return (int)(V.x[E_THREAD] + 0.5f); }
EXPORT float ni_tube_depth(void) { return V.x[E_DEPTH]; }
EXPORT float ni_tube_place(void) { return V.x[E_PLACE]; }
EXPORT float ni_tube_force(void) { return -V.x[E_AXIAL]; }
EXPORT int ni_tube_placed(int i) { return (int)V.D.episodes[i].nplaced; }
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
    snprintf(path, sizeof(path), "%s/scene_tube.bin", folder);
    if (FileExists(path)) {
        V.alt = V.D;
        memset(&V.D, 0, sizeof(V.D));
        load_scene(path);
        snprintf(path, sizeof(path), "%s/replays_tube.bin", folder);
        load_tube_replays(path);
        V.mode = 0;  // insertion first; alignment in V.alt
    } else {
        V.mode = 1;
    }
    V.pbr = LoadShaderFromMemory(ni_vs, ni_fs);
    V.depth = LoadShaderFromMemory(ni_vs, ni_depth_fs);
    const char* names[] = {"albedo", "surface", "camPos", "lightDir", "lightColor", "fillDir", "fillColor",
                           "exposure", "lightVP", "texel", "shadowBias", "opacity"};
    for (int k = 0; k <= L_OPACITY; k++) V.loc[k] = GetShaderLocation(V.pbr, names[k]);
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
    V.speed = V.D.kind == KIND_TUBE ? 0.0f : 1.0f;  // 0: automatic (insertion)
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
    int mode = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) folder = argv[++i];
        else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc) shot = argv[++i];
        else if (!strcmp(argv[i], "--episode") && i + 1 < argc) select = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i + 1 < argc) at = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--camera") && i + 1 < argc) camera = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--overlays") && i + 1 < argc) overlays = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--zoom") && i + 1 < argc) zoom = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 2 < argc) { width = atoi(argv[++i]); height = atoi(argv[++i]); }
    }
    init(folder, width, height);
    ni_mode(mode);
    V.overlays = overlays;
    ni_select(select);
    if (at >= 0) { ni_seek(at); ni_play(0); }
    sample(V.time);
    camera_preset(camera);
    V.distance *= zoom;
    V.zoomTo = V.distance;
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
