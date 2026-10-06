// Surgical alignment task core: MuJoCo stepping, programmed servo, disturbances
// and task logic. Shared by the PufferLib 5.0 adapter and the local test library,
// so training, evaluation and tests run the same physics and control code. SI units.
//
// The policy commands bounded needle-tip velocities at 50 Hz. A programmed servo
// (the C port of motion.Servo) turns the integrated reference into force-limited
// position-actuator commands at the 1 ms physics clock; it is the motor drive.
// Observations are simulator measurements corrupted by the disturbance layer;
// rewards and success use true simulator state. No camera pixels are used.
//
// Disturbance layer (brief v2), all magnitudes scaled by a level in [0, 1] and
// illustrative until replaced by measured hardware and tissue data:
//   forces    low-pass force noise and extra Coulomb friction on each slide;
//             table vibration as inertial forces through each carriage's mass
//   sensing   needle-tip measurement noise and latency; target estimate noise,
//             per-episode bias and slow drift
//   tissue    breathing and pulse motion of the true target (the rigid phantom's
//             collision surface is not moved in this stage)
// Level 0 reproduces the undisturbed task exactly: disturbance parameters come
// from a separate random stream, so start states and targets are unchanged.
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mujoco/mujoco.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SA_OBS 16
#define SA_ACT 3
#define SA_NJ 5
#define SA_NT 6
#define SA_REPEAT 20           // physics steps per policy step: 50 Hz policy
#define SA_HORIZON 250         // policy steps: 5 s deadline
#define SA_VMAX_XY 0.04        // m/s at |action| = 1
#define SA_VMAX_Z 0.02         // m/s at |action| = 1
#define SA_FILTER_S 0.01       // reference velocity smoothing time constant
#define SA_HOVER 0.001         // goal: 1 mm above the target site
#define SA_WORK_Z (-0.006)     // Z stage working height (motion.WORK_Z_M)
#define SA_CARRY 0.0005        // retainer carry extension (motion.CARRY_M)
#define SA_TOL 1e-5            // 10 µm lateral and vertical success tolerance
#define SA_SPEED_TOL 2e-4      // 0.2 mm/s, relative to the (moving) target
#define SA_HOLD_STEPS 15       // 0.3 s sustained
#define SA_INTEGRAL_TIME 0.05
#define SA_MAX_SEGMENTS 256
#define SA_VESSEL_HEIGHT 0.003 // vessel proximity applies when the tip is this low
#define SA_VESSEL_MARGIN 0.0003
#define SA_HISTORY 16          // physics steps of measurement history (latency)

// Full-strength (level 1) disturbance magnitudes.
#define SA_FORCE_TAU 0.05      // force-noise correlation time, s
static const double SA_FORCE_SIGMA[SA_NJ] = {0.10, 0.10, 0.05, 0.01, 0.002};   // N, stationary std
static const double SA_FRICTION_MAX[SA_NJ] = {0.10, 0.10, 0.05, 0.005, 0.001}; // N, extra Coulomb
static const double SA_AXIS[SA_NJ][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}, {0, 0, -1}};
#define SA_VIB_COMPONENTS 3
#define SA_VIB_AMP 0.01        // m/s^2 per component and axis
#define SA_VIB_FMIN 5.0
#define SA_VIB_FMAX 60.0
#define SA_TIP_NOISE 0.5e-6    // m
#define SA_VEL_NOISE 0.05e-3   // m/s
#define SA_LATENCY_MAX 10      // physics steps (ms)
#define SA_GOAL_NOISE 3e-6     // m per axis per observation
#define SA_GOAL_BIAS 1e-6      // m per axis per episode
#define SA_GOAL_DRIFT 1e-6     // m / sqrt(s)
#define SA_BREATH_Z 30e-6      // m
#define SA_BREATH_XY 10e-6
#define SA_PULSE_Z 10e-6
#define SA_PULSE_XY 3e-6

static const char* SA_JOINTS[SA_NJ] = {"stage_x", "stage_y", "stage_z", "insertion", "retainer"};

typedef struct {
    float success, collision, timeout, vessel_steps, final_lateral_um, final_vertical_um,
          episode_return, episode_length, level;
} SAEpisode;

typedef struct {
    double level;
    double force[SA_NJ], friction[SA_NJ];
    double vib_amp[3][SA_VIB_COMPONENTS], vib_freq[3][SA_VIB_COMPONENTS], vib_phase[3][SA_VIB_COMPONENTS];
    double motion_amp[2][3], motion_freq[2], motion_phase[2];  // breathing, pulse
    double goal_bias[3], goal_drift[3];
    int latency;
} SADisturbance;

typedef struct {
    const mjModel* m;
    mjData* d;
    int qadr[SA_NJ], dof[SA_NJ], act[SA_NJ];
    double kp[SA_NJ], kv[SA_NJ], damping[SA_NJ], friction[SA_NJ], ki[SA_NJ], mass[SA_NJ];
    double lo[SA_NJ], hi[SA_NJ], clo[SA_NJ], chi[SA_NJ];
    double integral[SA_NJ], q_ref[SA_NJ], v_ref[SA_NJ], v_cmd[SA_NJ];
    double tip0[3], targets[SA_NT][3], goal0[3], goal[3], goal_vel[3];
    double vessels[SA_MAX_SEGMENTS][5];
    int nvessel;
    double dome[5];
    int tip_site;
    unsigned char* robot_geom;
    int target, tick, hold, vessel_steps, collided;
    float prev_action[SA_ACT];
    double potential, episode_return;
    uint64_t rng, drng;
    double level_max;
    int level_mode;  // 0: every episode at level_max; 1: uniform in [0, level_max]
    SADisturbance dist;
    double hist_tip[SA_HISTORY][3], hist_vel[SA_HISTORY][3];
    int hist_count;
    // Diagnostics for replay overlays (last physics step / last observation).
    double dbg_force[SA_NJ], dbg_friction[SA_NJ], dbg_inertial[SA_NJ], dbg_base_acc[3];
    double dbg_meas_tip[3], dbg_meas_goal[3];
    double acc[16], inertial[16];
} SACore;

static inline uint64_t sa_next(uint64_t* s) {  // splitmix64
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline double sa_uniform(uint64_t* s, double a, double b) {
    return a + (b - a) * ((sa_next(s) >> 11) * (1.0 / 9007199254740992.0));
}

static inline double sa_normal(uint64_t* s) {  // Box-Muller
    double u = sa_uniform(s, 1e-12, 1.0), v = sa_uniform(s, 0.0, 1.0);
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

static inline double sa_clip(double x, double a, double b) {
    return x < a ? a : (x > b ? b : x);
}

static inline double sa_surface(const SACore* c, double x, double y) {
    double u = x / c->dome[3], v = (y - c->dome[1]) / c->dome[4];
    return c->dome[0] - c->dome[2] * (u * u + v * v);
}

// Nearest vessel centerline point to (x, y); returns its distance minus radius.
static inline double sa_vessel(const SACore* c, double x, double y, double* dx, double* dy) {
    double best = 1e9;
    *dx = *dy = 0;
    for (int i = 0; i < c->nvessel; i++) {
        const double* s = c->vessels[i];
        double ex = s[2] - s[0], ey = s[3] - s[1];
        double len2 = ex * ex + ey * ey;
        double t = len2 > 0 ? sa_clip(((x - s[0]) * ex + (y - s[1]) * ey) / len2, 0, 1) : 0;
        double px = s[0] + t * ex - x, py = s[1] + t * ey - y;
        double dist = sqrt(px * px + py * py) - s[4];
        if (dist < best) {
            best = dist;
            *dx = px;
            *dy = py;
        }
    }
    return best;
}

static inline void sa_tip(const SACore* c, double* tip, double* vel) {
    memcpy(tip, c->d->site_xpos + 3 * c->tip_site, 3 * sizeof(double));
    // Cartesian gantry: tip velocity follows the joint velocities directly.
    vel[0] = c->d->qvel[c->dof[0]];
    vel[1] = c->d->qvel[c->dof[1]];
    vel[2] = c->d->qvel[c->dof[2]] - c->d->qvel[c->dof[3]];
}

static inline const mjModel* sa_load_model(const char* path) {
    static const mjModel* shared = NULL;  // read-only, shared by every world
    static char loaded[1024] = {0};
    if (shared && strcmp(loaded, path) == 0) {
        return shared;
    }
    char error[1000] = "";
    mjModel* m = mj_loadXML(path, NULL, error, sizeof(error));
    if (!m) {
        fprintf(stderr, "surgical_align: cannot load %s: %s\n", path, error);
        exit(1);
    }
    shared = m;
    snprintf(loaded, sizeof(loaded), "%s", path);
    return shared;
}

static inline int sa_init(SACore* c, const char* path, uint64_t seed) {
    memset(c, 0, sizeof(*c));
    c->m = sa_load_model(path);
    const mjModel* m = c->m;
    if (m->nv > 16) {
        fprintf(stderr, "surgical_align: unexpected model size\n");
        exit(1);
    }
    c->d = mj_makeData(m);
    for (int j = 0; j < SA_NJ; j++) {
        int jid = mj_name2id(m, mjOBJ_JOINT, SA_JOINTS[j]);
        int aid = mj_name2id(m, mjOBJ_ACTUATOR, SA_JOINTS[j]);
        if (jid < 0 || aid < 0) {
            fprintf(stderr, "surgical_align: missing joint/actuator %s\n", SA_JOINTS[j]);
            exit(1);
        }
        c->qadr[j] = m->jnt_qposadr[jid];
        c->dof[j] = m->jnt_dofadr[jid];
        c->act[j] = aid;
        c->kp[j] = m->actuator_gainprm[aid * mjNGAIN];
        c->kv[j] = -m->actuator_biasprm[aid * mjNBIAS + 2];
        c->damping[j] = m->dof_damping[c->dof[j]];
        c->friction[j] = m->dof_frictionloss[c->dof[j]];
        c->ki[j] = c->kp[j] / SA_INTEGRAL_TIME;
        c->lo[j] = m->jnt_range[2 * jid];
        c->hi[j] = m->jnt_range[2 * jid + 1];
        c->clo[j] = m->actuator_ctrlrange[2 * aid];
        c->chi[j] = m->actuator_ctrlrange[2 * aid + 1];
    }
    c->tip_site = mj_name2id(m, mjOBJ_SITE, "needle_tip");
    mj_resetData(m, c->d);
    mj_forward(m, c->d);
    memcpy(c->tip0, c->d->site_xpos + 3 * c->tip_site, 3 * sizeof(double));
    // Effective mass moved along each slide (diagonal of the mass matrix).
    for (int j = 0; j < SA_NJ; j++) {
        memset(c->acc, 0, sizeof(c->acc));
        c->acc[c->dof[j]] = 1;
        mj_mulM(m, c->d, c->inertial, c->acc);
        c->mass[j] = c->inertial[c->dof[j]];
    }
    for (int k = 0; k < SA_NT; k++) {
        char name[32];
        snprintf(name, sizeof(name), "target_%d", k);
        int sid = mj_name2id(m, mjOBJ_SITE, name);
        memcpy(c->targets[k], c->d->site_xpos + 3 * sid, 3 * sizeof(double));
    }
    int dome = mj_name2id(m, mjOBJ_NUMERIC, "surface_dome");
    memcpy(c->dome, m->numeric_data + m->numeric_adr[dome], 5 * sizeof(double));
    for (int i = 0; i < SA_MAX_SEGMENTS; i++) {
        char name[32];
        snprintf(name, sizeof(name), "vessel_%03d", i);
        int nid = mj_name2id(m, mjOBJ_NUMERIC, name);
        if (nid < 0) {
            break;
        }
        memcpy(c->vessels[i], m->numeric_data + m->numeric_adr[nid], 5 * sizeof(double));
        c->nvessel++;
    }
    c->robot_geom = (unsigned char*)calloc(m->ngeom, 1);
    const char* bodies[] = {"stage_x", "stage_y", "stage_z", "insertion", "retainer"};
    for (int g = 0; g < m->ngeom; g++) {
        for (int b = 0; b < 5; b++) {
            if (m->geom_bodyid[g] == mj_name2id(m, mjOBJ_BODY, bodies[b])) {
                c->robot_geom[g] = 1;
            }
        }
    }
    c->rng = seed * 0x2545F4914F6CDD1DULL + 1;
    c->drng = (seed ^ 0xD157AB0CE5ULL) * 0x9E3779B97F4A7C15ULL + 7;  // independent disturbance stream
    return 0;
}

// Disturbance strength: level_max in [0, 1]; mode 0 fixes every episode at
// level_max, mode 1 draws each episode's level uniformly in [0, level_max].
static inline void sa_set_disturbance(SACore* c, double level_max, int mode) {
    c->level_max = sa_clip(level_max, 0, 1);
    c->level_mode = mode;
}

static inline void sa_free(SACore* c) {
    if (c->d) {
        mj_deleteData(c->d);
    }
    free(c->robot_geom);
    c->d = NULL;
    c->robot_geom = NULL;
}

static inline void sa_draw_disturbance(SACore* c) {
    SADisturbance* q = &c->dist;
    memset(q, 0, sizeof(*q));
    uint64_t* r = &c->drng;
    double level = c->level_mode ? sa_uniform(r, 0, c->level_max) : c->level_max;
    q->level = level;
    if (level <= 0) {
        return;
    }
    for (int j = 0; j < SA_NJ; j++) {
        q->force[j] = SA_FORCE_SIGMA[j] * level * sa_normal(r);  // start in the stationary distribution
        q->friction[j] = SA_FRICTION_MAX[j] * level * sa_uniform(r, 0, 1);
    }
    for (int a = 0; a < 3; a++) {
        for (int k = 0; k < SA_VIB_COMPONENTS; k++) {
            q->vib_amp[a][k] = SA_VIB_AMP * level * sa_uniform(r, 0, 1);
            q->vib_freq[a][k] = sa_uniform(r, SA_VIB_FMIN, SA_VIB_FMAX);
            q->vib_phase[a][k] = sa_uniform(r, 0, 2 * M_PI);
        }
    }
    const double freq_lo[2] = {0.2, 1.0}, freq_hi[2] = {0.4, 2.0};
    const double amp_z[2] = {SA_BREATH_Z, SA_PULSE_Z}, amp_xy[2] = {SA_BREATH_XY, SA_PULSE_XY};
    for (int k = 0; k < 2; k++) {
        double heading = sa_uniform(r, 0, 2 * M_PI), lateral = amp_xy[k] * level * sa_uniform(r, 0, 1);
        q->motion_amp[k][0] = lateral * cos(heading);
        q->motion_amp[k][1] = lateral * sin(heading);
        q->motion_amp[k][2] = amp_z[k] * level * sa_uniform(r, 0, 1);
        q->motion_freq[k] = sa_uniform(r, freq_lo[k], freq_hi[k]);
        q->motion_phase[k] = sa_uniform(r, 0, 2 * M_PI);
    }
    for (int a = 0; a < 3; a++) {
        q->goal_bias[a] = SA_GOAL_BIAS * level * sa_normal(r);
    }
    q->latency = (int)floor(sa_uniform(r, 0, SA_LATENCY_MAX * level + 1));
    if (q->latency > SA_LATENCY_MAX) q->latency = SA_LATENCY_MAX;
}

// True target (moving with breathing and pulse) and its velocity at time t.
static inline void sa_update_goal(SACore* c, double t) {
    const SADisturbance* q = &c->dist;
    for (int a = 0; a < 3; a++) {
        c->goal[a] = c->goal0[a];
        c->goal_vel[a] = 0;
        for (int k = 0; k < 2; k++) {
            double w = 2 * M_PI * q->motion_freq[k];
            c->goal[a] += q->motion_amp[k][a] * sin(w * t + q->motion_phase[k]);
            c->goal_vel[a] += q->motion_amp[k][a] * w * cos(w * t + q->motion_phase[k]);
        }
    }
}

static inline double sa_distance(const SACore* c, double* lateral, double* vertical) {
    const double* tip = c->d->site_xpos + 3 * c->tip_site;
    double dx = tip[0] - c->goal[0], dy = tip[1] - c->goal[1], dz = tip[2] - c->goal[2];
    *lateral = sqrt(dx * dx + dy * dy);
    *vertical = fabs(dz);
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// Potential for shaping: log-distance gives resolution from centimetres to microns.
static inline double sa_potential(double distance) {
    return -log10((distance + 2e-6) / 1e-3);
}

static inline void sa_record(SACore* c) {
    double tip[3], vel[3];
    sa_tip(c, tip, vel);
    int slot = c->hist_count % SA_HISTORY;
    memcpy(c->hist_tip[slot], tip, sizeof(tip));
    memcpy(c->hist_vel[slot], vel, sizeof(vel));
    c->hist_count++;
}

// Measured tip state: delayed by the episode's latency and corrupted by sensor noise.
static inline void sa_measure(SACore* c, double* tip, double* vel) {
    sa_tip(c, tip, vel);
    int lag = c->dist.latency;
    if (lag > 0 && c->hist_count > 0) {
        int back = lag <= c->hist_count ? lag : c->hist_count;
        int slot = (c->hist_count - back) % SA_HISTORY;
        memcpy(tip, c->hist_tip[slot], 3 * sizeof(double));
        memcpy(vel, c->hist_vel[slot], 3 * sizeof(double));
    }
    if (c->dist.level > 0) {
        for (int a = 0; a < 3; a++) {
            tip[a] += SA_TIP_NOISE * c->dist.level * sa_normal(&c->drng);
            vel[a] += SA_VEL_NOISE * c->dist.level * sa_normal(&c->drng);
        }
    }
}

static inline void sa_observe(SACore* c, float* obs) {
    double tip[3], vel[3], goal[3];
    sa_measure(c, tip, vel);
    for (int a = 0; a < 3; a++) {
        goal[a] = c->goal[a];
        if (c->dist.level > 0) {
            goal[a] += c->dist.goal_bias[a] + c->dist.goal_drift[a] + SA_GOAL_NOISE * c->dist.level * sa_normal(&c->drng);
        }
    }
    memcpy(c->dbg_meas_tip, tip, sizeof(tip));
    memcpy(c->dbg_meas_goal, goal, sizeof(goal));
    for (int i = 0; i < 3; i++) {
        double e = tip[i] - goal[i];
        obs[i] = (float)sa_clip(e / 0.005, -5, 5);
        obs[3 + i] = (float)sa_clip(vel[i] / SA_VMAX_XY, -2, 2);
        obs[6 + i] = c->prev_action[i];
        obs[9 + i] = (float)sa_clip(e / 1e-4, -1, 1);
    }
    double vx, vy;
    sa_vessel(c, tip[0], tip[1], &vx, &vy);
    obs[12] = (float)sa_clip(vx / 0.005, -1, 1);
    obs[13] = (float)sa_clip(vy / 0.005, -1, 1);
    obs[14] = (float)sa_clip((tip[2] - sa_surface(c, tip[0], tip[1])) / 0.01, -1, 2);
    obs[15] = (float)(1.0 - (double)c->tick / SA_HORIZON);
}

// Initialization only: place the gantry at rest at a random start, choose a target.
static inline void sa_reset(SACore* c, float* obs) {
    const mjModel* m = c->m;
    mjData* d = c->d;
    mj_resetData(m, d);
    double x = sa_uniform(&c->rng, -0.04, 0.04), y = sa_uniform(&c->rng, -0.03, 0.03);
    double wx = c->tip0[0] + x, wy = c->tip0[1] + y;
    double tip_z = sa_surface(c, wx, wy) + sa_uniform(&c->rng, 0.003, 0.008);
    double q[SA_NJ] = {x, y, SA_WORK_Z, SA_WORK_Z - (tip_z - c->tip0[2]), SA_CARRY};
    for (int j = 0; j < SA_NJ; j++) {
        d->qpos[c->qadr[j]] = q[j];
        c->q_ref[j] = q[j];
        c->v_ref[j] = c->v_cmd[j] = c->integral[j] = 0;
    }
    mj_forward(m, d);
    c->target = (int)(sa_next(&c->rng) % SA_NT);
    for (int i = 0; i < 3; i++) {
        c->goal0[i] = c->targets[c->target][i];
    }
    c->goal0[2] += SA_HOVER;
    sa_draw_disturbance(c);
    sa_update_goal(c, 0.0);
    c->tick = c->hold = c->vessel_steps = c->collided = 0;
    c->hist_count = 0;
    c->episode_return = 0;
    memset(c->prev_action, 0, sizeof(c->prev_action));
    double lat, vert;
    c->potential = sa_potential(sa_distance(c, &lat, &vert));
    sa_observe(c, obs);
}

// Programmed servo, after mj_step1: PD from the actuator plus feedforward and a
// bounded integral, realized by offsetting the position command. It reads the
// drive's own (exact) joint encoders, as a stage controller would.
static inline void sa_servo(SACore* c, const double* a_ref) {
    const mjModel* m = c->m;
    mjData* d = c->d;
    double dt = m->opt.timestep;
    memset(c->acc, 0, sizeof(c->acc));
    for (int j = 0; j < SA_NJ; j++) {
        c->acc[c->dof[j]] = a_ref[j];
    }
    mj_mulM(m, d, c->inertial, c->acc);
    for (int j = 0; j < SA_NJ; j++) {
        double e = c->q_ref[j] - d->qpos[c->qadr[j]];
        double ff = c->inertial[c->dof[j]] + d->qfrc_bias[c->dof[j]] + c->damping[j] * c->v_ref[j]
            + c->friction[j] * tanh(c->v_ref[j] / 1e-4);
        double limit = 2 * c->friction[j] + 1e-12;
        c->integral[j] = sa_clip(c->integral[j] + c->ki[j] * e * dt, -limit, limit);
        double ctrl = c->q_ref[j] + (c->kv[j] * c->v_ref[j] + ff + c->integral[j]) / c->kp[j];
        d->ctrl[c->act[j]] = sa_clip(ctrl, c->clo[j], c->chi[j]);
    }
}

// External generalized forces on the slides for this physics step: correlated
// force noise, extra Coulomb friction, and table vibration as inertial forces.
static inline void sa_disturb(SACore* c) {
    mjData* d = c->d;
    SADisturbance* q = &c->dist;
    for (int j = 0; j < SA_NJ; j++) {
        d->qfrc_applied[c->dof[j]] = 0;
        c->dbg_force[j] = c->dbg_friction[j] = c->dbg_inertial[j] = 0;
    }
    c->dbg_base_acc[0] = c->dbg_base_acc[1] = c->dbg_base_acc[2] = 0;
    if (q->level <= 0) {
        return;
    }
    double dt = c->m->opt.timestep, t = d->time, a_base[3] = {0, 0, 0};
    for (int a = 0; a < 3; a++) {
        for (int k = 0; k < SA_VIB_COMPONENTS; k++) {
            a_base[a] += q->vib_amp[a][k] * sin(2 * M_PI * q->vib_freq[a][k] * t + q->vib_phase[a][k]);
        }
    }
    for (int j = 0; j < SA_NJ; j++) {
        double sigma = SA_FORCE_SIGMA[j] * q->level;
        q->force[j] += -q->force[j] * dt / SA_FORCE_TAU + sigma * sqrt(2 * dt / SA_FORCE_TAU) * sa_normal(&c->drng);
        double v = d->qvel[c->dof[j]];
        double inertial = -c->mass[j] * (a_base[0] * SA_AXIS[j][0] + a_base[1] * SA_AXIS[j][1] + a_base[2] * SA_AXIS[j][2]);
        double friction = -q->friction[j] * tanh(v / 1e-4);
        d->qfrc_applied[c->dof[j]] = q->force[j] + friction + inertial;
        c->dbg_force[j] = q->force[j];
        c->dbg_friction[j] = friction;
        c->dbg_inertial[j] = inertial;
    }
    memcpy(c->dbg_base_acc, a_base, sizeof(a_base));
}

// Cubic action map gives fine velocity resolution near zero.
static inline void sa_command(SACore* c, const float* action) {
    double a[SA_ACT];
    for (int i = 0; i < SA_ACT; i++) {
        a[i] = sa_clip(action[i], -1, 1);
        a[i] = a[i] * a[i] * a[i];
    }
    double vx = a[0] * SA_VMAX_XY, vy = a[1] * SA_VMAX_XY, vz = a[2] * SA_VMAX_Z;
    double v[SA_NJ] = {vx, vy, 0, -vz, 0};  // insertion axis is world -Z
    memcpy(c->v_cmd, v, sizeof(v));
}

static inline int sa_contact(const SACore* c) {
    const mjData* d = c->d;
    for (int i = 0; i < d->ncon; i++) {
        if (c->robot_geom[d->contact[i].geom[0]] != c->robot_geom[d->contact[i].geom[1]]) {
            return 1;
        }
    }
    return 0;
}

// One policy step. Returns 1 when the episode ended and fills `episode`.
static inline int sa_step(SACore* c, const float* action, float* obs, float* reward, SAEpisode* episode) {
    const mjModel* m = c->m;
    mjData* d = c->d;
    double dt = m->opt.timestep, a_ref[SA_NJ];
    sa_command(c, action);
    int contact = 0;
    for (int s = 0; s < SA_REPEAT; s++) {
        mj_step1(m, d);
        sa_record(c);
        for (int j = 0; j < SA_NJ; j++) {
            double next = c->v_ref[j] + (c->v_cmd[j] - c->v_ref[j]) * dt / SA_FILTER_S;
            double q = c->q_ref[j] + next * dt;
            // Keep the reference inside travel with a 0.2 mm margin.
            double qmin = c->lo[j] + 2e-4, qmax = c->hi[j] - 2e-4;
            if (q < qmin || q > qmax) {
                q = sa_clip(q, qmin, qmax);
                next = 0;
            }
            a_ref[j] = (next - c->v_ref[j]) / dt;
            c->v_ref[j] = next;
            c->q_ref[j] = q;
        }
        sa_servo(c, a_ref);
        sa_disturb(c);
        contact |= sa_contact(c);
        mj_step2(m, d);
    }
    mj_kinematics(m, d);
    c->tick++;
    if (c->dist.level > 0) {
        double step = SA_REPEAT * dt;
        for (int a = 0; a < 3; a++) {
            c->dist.goal_drift[a] += SA_GOAL_DRIFT * c->dist.level * sqrt(step) * sa_normal(&c->drng);
        }
    }
    sa_update_goal(c, d->time);
    double tip[3], vel[3], lat, vert;
    sa_tip(c, tip, vel);
    double distance = sa_distance(c, &lat, &vert);
    double potential = sa_potential(distance);
    double r = potential - c->potential;
    c->potential = potential;
    // Penalize changes of the applied (clipped) action. Raw policy outputs are
    // unbounded Gaussian samples; penalizing them diverged the first run.
    double smooth = 0;
    for (int i = 0; i < SA_ACT; i++) {
        double applied = sa_clip(action[i], -1, 1);
        double da = applied - c->prev_action[i];
        smooth += da * da;
        c->prev_action[i] = (float)applied;
    }
    r -= 0.002 * smooth;
    double vx, vy;
    double clearance = sa_vessel(c, tip[0], tip[1], &vx, &vy);
    if (tip[2] - sa_surface(c, tip[0], tip[1]) < SA_VESSEL_HEIGHT && clearance < SA_VESSEL_MARGIN) {
        r -= 0.05;
        c->vessel_steps++;
    }
    double rel[3] = {vel[0] - c->goal_vel[0], vel[1] - c->goal_vel[1], vel[2] - c->goal_vel[2]};
    double speed = sqrt(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
    c->hold = (lat < SA_TOL && vert < SA_TOL && speed < SA_SPEED_TOL) ? c->hold + 1 : 0;
    int finite = 1;
    for (int i = 0; i < m->nq; i++) {
        finite &= isfinite(d->qpos[i]);
    }
    int done = 0;
    float success = 0, collision = 0, timeout = 0;
    if (contact || !finite) {
        r -= 2;
        done = 1;
        collision = 1;
    } else if (c->hold >= SA_HOLD_STEPS) {
        r += 2;
        done = 1;
        success = 1;
    } else if (c->tick >= SA_HORIZON) {
        done = 1;
        timeout = 1;
    }
    *reward = (float)r;
    c->episode_return += r;
    if (done && episode) {
        episode->success = success;
        episode->collision = collision;
        episode->timeout = timeout;
        episode->vessel_steps = (float)c->vessel_steps;
        episode->final_lateral_um = (float)(lat * 1e6);
        episode->final_vertical_um = (float)(vert * 1e6);
        episode->episode_return = (float)c->episode_return;
        episode->episode_length = (float)c->tick;
        episode->level = (float)c->dist.level;
    }
    sa_observe(c, obs);
    return done;
}

// Scripted reference controller on the same observation: proportional tip
// velocity toward the goal, inverted through the cubic action map. Until the tip
// is within 0.5 mm laterally it keeps at least 3 mm above the dome, so it neither
// cuts through the phantom nor passes low over vessels. Programmed control, used
// as a yardstick for learning, not a deployed component.
static inline void sa_scripted(const float* obs, float* action) {
    const double gain = 6.0;  // 1/s
    double vmax[3] = {SA_VMAX_XY, SA_VMAX_XY, SA_VMAX_Z}, e[3], v[3];
    for (int i = 0; i < 3; i++) {
        e[i] = fabs(obs[9 + i]) < 1 ? obs[9 + i] * 1e-4 : obs[i] * 0.005;
        v[i] = -gain * e[i];
    }
    double lateral = sqrt(e[0] * e[0] + e[1] * e[1]);
    double height = obs[14] * 0.01;
    if (lateral > 5e-4 && gain * (SA_VESSEL_HEIGHT - height) > v[2]) {
        v[2] = gain * (SA_VESSEL_HEIGHT - height);
    }
    for (int i = 0; i < 3; i++) {
        action[i] = (float)cbrt(sa_clip(v[i], -vmax[i], vmax[i]) / vmax[i]);
    }
}

// Replay row (SA_ROW doubles): true tip(3), true goal(3), joints(5), time, target,
// |action|, level, measured tip(3), measured goal(3), table acceleration(3),
// force noise(5), extra friction(5), vibration inertial force(5), latency (s).
#define SA_ROW 40
static inline void sa_replay_row(const SACore* c, double action_norm, double* r) {
    memcpy(r, c->d->site_xpos + 3 * c->tip_site, 3 * sizeof(double));
    memcpy(r + 3, c->goal, 3 * sizeof(double));
    for (int j = 0; j < SA_NJ; j++) {
        r[6 + j] = c->d->qpos[c->qadr[j]];
    }
    r[11] = c->d->time;
    r[12] = c->target;
    r[13] = action_norm;
    r[14] = c->dist.level;
    memcpy(r + 15, c->dbg_meas_tip, 3 * sizeof(double));
    memcpy(r + 18, c->dbg_meas_goal, 3 * sizeof(double));
    memcpy(r + 21, c->dbg_base_acc, 3 * sizeof(double));
    memcpy(r + 24, c->dbg_force, SA_NJ * sizeof(double));
    memcpy(r + 29, c->dbg_friction, SA_NJ * sizeof(double));
    memcpy(r + 34, c->dbg_inertial, SA_NJ * sizeof(double));
    r[39] = c->dist.latency * c->m->opt.timestep;
}
