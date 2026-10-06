// Surgical alignment task core: MuJoCo stepping, programmed servo, task logic.
// Shared by the PufferLib 5.0 adapter and the local test library, so training,
// evaluation and tests run the same physics and control code. SI units.
//
// The policy commands bounded needle-tip velocities at 50 Hz. A programmed servo
// (the C port of motion.Servo) turns the integrated reference into force-limited
// position-actuator commands at the 1 ms physics clock. Observations use exact
// simulator state (labelled as such); no camera pixels are used.
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mujoco/mujoco.h>

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
#define SA_SPEED_TOL 2e-4      // 0.2 mm/s
#define SA_HOLD_STEPS 15       // 0.3 s sustained
#define SA_INTEGRAL_TIME 0.05
#define SA_MAX_SEGMENTS 256
#define SA_VESSEL_HEIGHT 0.003 // vessel proximity applies when the tip is this low
#define SA_VESSEL_MARGIN 0.0003

static const char* SA_JOINTS[SA_NJ] = {"stage_x", "stage_y", "stage_z", "insertion", "retainer"};

typedef struct {
    float success, collision, timeout, vessel_steps, final_lateral_um, final_vertical_um,
          episode_return, episode_length;
} SAEpisode;

typedef struct {
    const mjModel* m;
    mjData* d;
    int qadr[SA_NJ], dof[SA_NJ], act[SA_NJ];
    double kp[SA_NJ], kv[SA_NJ], damping[SA_NJ], friction[SA_NJ], ki[SA_NJ];
    double lo[SA_NJ], hi[SA_NJ], clo[SA_NJ], chi[SA_NJ];
    double integral[SA_NJ], q_ref[SA_NJ], v_ref[SA_NJ], v_cmd[SA_NJ];
    double tip0[3], targets[SA_NT][3], goal[3];
    double vessels[SA_MAX_SEGMENTS][5];
    int nvessel;
    double dome[5];
    int tip_site;
    unsigned char* robot_geom;
    int target, tick, hold, vessel_steps, collided;
    float prev_action[SA_ACT];
    double potential, episode_return;
    uint64_t rng;
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
    mj_kinematics(m, c->d);
    memcpy(c->tip0, c->d->site_xpos + 3 * c->tip_site, 3 * sizeof(double));
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
    return 0;
}

static inline void sa_free(SACore* c) {
    if (c->d) {
        mj_deleteData(c->d);
    }
    free(c->robot_geom);
    c->d = NULL;
    c->robot_geom = NULL;
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

static inline void sa_observe(SACore* c, float* obs) {
    double tip[3], vel[3], lat, vert;
    sa_tip(c, tip, vel);
    sa_distance(c, &lat, &vert);
    for (int i = 0; i < 3; i++) {
        double e = tip[i] - c->goal[i];
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
        c->goal[i] = c->targets[c->target][i];
    }
    c->goal[2] += SA_HOVER;
    c->tick = c->hold = c->vessel_steps = c->collided = 0;
    c->episode_return = 0;
    memset(c->prev_action, 0, sizeof(c->prev_action));
    double lat, vert;
    c->potential = sa_potential(sa_distance(c, &lat, &vert));
    sa_observe(c, obs);
}

// Programmed servo, after mj_step1: PD from the actuator plus feedforward and a
// bounded integral, realized by offsetting the position command.
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
        contact |= sa_contact(c);
        mj_step2(m, d);
    }
    mj_kinematics(m, d);
    c->tick++;
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
    double speed = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
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
    }
    sa_observe(c, obs);
    return done;
}

// Scripted reference controller on the same observation: proportional tip
// velocity toward the goal, inverted through the cubic action map. Until the tip
// is within 0.5 mm laterally it keeps at least 3 mm above the dome, so it neither
// cuts through the phantom nor passes low over vessels. Programmed control, used
// as a feasibility baseline, not a learned policy.
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
