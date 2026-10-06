// Thread insertion task core (thread-tube design), built on the alignment core (surgical_core.h): the same
// robot model, programmed servo, disturbance layer and measurement model. Shared by the PufferLib 5.0
// adapter for training and by the local ctypes library for tests, yardsticks and evaluation. SI units.
//
// One site per episode. The phantom is turned about its dome centre by a random angle (the vessel layout
// turns with it; the dome shape is kept); the site is random on the dome, clear of vessels and of 0-5
// threads already standing at earlier sites. The policy steers the needle point (stage X, Y and Z
// velocities, as in alignment) and has a fourth action that starts the programmed stroke of the approved
// cycle (tube_cycle.py), with the stages held still from then on: the insertion slide drives the needle
// 1.02 mm down onto the waiting thread end in 40 ms, which sticks to it (a stand-in for a chemical bond),
// waits 5 ms, drives on to 2 mm below the measured surface in 25 ms, waits 30 ms, lets go, waits 10 ms
// and snaps back in 30 ms.
//
// The thread is not simulated here (it is in the MuJoCo cycle, tube_cycle.py, used to validate). Its end
// waits 1 mm below the needle point, offset from the needle's axis by an amount drawn per episode from the
// full simulation's statistics, and jumps sideways at release by a drawn amount. The robot measures the
// offset (with noise) as a camera would. The tissue is a needle-tissue force model (puncture, cutting,
// shaft friction, lateral spring) that moves with breathing and pulse; collisions with the phantom are
// MuJoCo contacts for every robot part except the needle, which enters the tissue.
//
// Failures end the episode: robot contact with the phantom or fixtures, the needle or thread tube
// touching a placed thread, the waiting thread end dragged into the tissue, the needle missing the
// thread end, a nonfinite state. Success: the released thread end within 10 µm of the target.
#pragma once
#include "surgical_core.h"

#define SI_OBS 24
#define SI_ACT 4
#define SI_HORIZON 300            // policy steps: 6 s
#define SI_Q_READY 0.006          // insertion slide position with the needle ready above the thread end
#define SI_END_BELOW 0.001        // the waiting thread end, below the needle point
#define SI_READY 0.002            // hover: needle point this far above the surface (the end 1 mm above)
#define SI_DEPTH 0.002            // the thread end's depth at full stroke
#define SI_MARK 0.00016           // target markings sit this far above the surface
#define SI_STICK 30e-6            // the end sticks if it is within this of the needle's axis
// Programmed stroke, minimum-jerk segments timed as in tube_cycle.py (pick, insert, release, snap back).
#define SI_PICK_S 0.040           // needle down onto the thread end
#define SI_PICK_PAST 20e-6        // the pick ends this far below the thread end
#define SI_PICK_HOLD_S 0.005
#define SI_INSERT_S 0.025         // on to depth
#define SI_STROKE_S (SI_PICK_S + SI_PICK_HOLD_S + SI_INSERT_S)
#define SI_SETTLE_S 0.030         // at depth before release
#define SI_RELEASE_S 0.010        // after release, before the snap back
#define SI_RETRACT_S 0.030
#define SI_TOL 10e-6              // success: thread end within 10 µm of the target
#define SI_MAX_PLACED 5
#define SI_THREAD_H 0.0062        // placed threads stand this far out of the tissue
#define SI_THREAD_R 20e-6
#define SI_SITE_VESSEL 0.0005     // a site keeps this clearance from vessels (beyond their radius)
#define SI_SITE_THREAD 0.0025     // and from placed threads
#define SI_SITE_SCALE 0.6         // sites lie within this fraction of the dome's semi-axes
#define SI_TUBE_SIN 0.25881904510 // thread tube, 15 degrees from vertical toward +X
#define SI_TUBE_COS 0.96592582629
#define SI_TUBE_EXIT 0.34e-3      // tube exit, along the tube from the waiting thread end
#define SI_TUBE_L 0.009
#define SI_TUBE_R 0.14e-3
#define SI_NEEDLE_R 50e-6
#define SI_NEEDLE_L 0.015
#define SI_END_NOISE 2e-6         // thread end measurement noise per observation, per axis
// Thread-end statistics from the full simulation (outputs/neural_insertion/thread_stats, training seeds
// 1-60, random site orders, levels 0-2): offset of the end from the needle's axis when the needle reaches
// it, and the end's sideways jump at release. Set by insert_core_stats.h, generated from those runs.
#include "insert_core_stats.h"
// Needle-tissue model (tissue.py, SI).
#define SI_K_DIMPLE 5.0
#define SI_D_PUNCTURE 0.0004
#define SI_F_CUT 0.0005
#define SI_F_SHAFT 0.2            // N per metre of inserted shaft
#define SI_V_SMOOTH 1e-4
#define SI_K_LATERAL 30.0
#define SI_C_LATERAL 0.05

enum { SI_APPROACH = 0, SI_STROKE = 1, SI_SETTLE = 2, SI_RETRACT = 3, SI_DONE = 4 };

typedef struct {
    float success, collision, thread_touch, dragged, missed, timeout, placement_um, depth_um, vessel_steps,
          episode_return, episode_length, level, nplaced;
} SIEpisode;

typedef struct {
    SACore a;
    double base_vessels[SA_MAX_SEGMENTS][5];
    double theta;
    double site[3];               // target marking, tissue at rest (world)
    double delta[2], delta_meas[2], jump[2];
    double placed[SI_MAX_PLACED][2];
    int nplaced;
    int phase;
    double phase_t, stroke_from, stroke_len, retract_from;
    int bonded, missed, dragged, thread_touch, punctured, entered;
    double entry[2];              // needle entry point (tissue frame)
    double end[3];                // thread end (world)
    double end_tissue[3];         // released thread end, tissue frame
    double placement, depth;
    double potential, episode_return;
    int tick, vessel_steps, triggered;
    float prev_action[SI_ACT];
    int needle_body;
} SICore;

// ---------------------------------------------------------------- geometry

static inline double si_seg_dist(const double* p0, const double* p1, const double* q0, const double* q1) {
    // Closest distance between segments p0-p1 and q0-q1.
    double d1[3], d2[3], r[3];
    for (int k = 0; k < 3; k++) { d1[k] = p1[k] - p0[k]; d2[k] = q1[k] - q0[k]; r[k] = p0[k] - q0[k]; }
    double a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2], e = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
    double f = d2[0] * r[0] + d2[1] * r[1] + d2[2] * r[2];
    double s, t;
    if (a < 1e-18 && e < 1e-18) { s = t = 0; }
    else if (a < 1e-18) { s = 0; t = sa_clip(f / e, 0, 1); }
    else {
        double c = d1[0] * r[0] + d1[1] * r[1] + d1[2] * r[2];
        if (e < 1e-18) { t = 0; s = sa_clip(-c / a, 0, 1); }
        else {
            double b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2], den = a * e - b * b;
            s = den > 1e-24 ? sa_clip((b * f - c * e) / den, 0, 1) : 0;
            t = (b * s + f) / e;
            if (t < 0) { t = 0; s = sa_clip(-c / a, 0, 1); }
            else if (t > 1) { t = 1; s = sa_clip((b - c) / a, 0, 1); }
        }
    }
    double dx = 0;
    for (int k = 0; k < 3; k++) { double v = r[k] + d1[k] * s - d2[k] * t; dx += v * v; }
    return sqrt(dx);
}

static inline void si_tissue(const SICore* c, double* off) {
    for (int k = 0; k < 3; k++) off[k] = c->a.goal[k] - c->a.goal0[k];  // the target moves with the tissue
}

// Tissue surface under (x, y), with the tissue displaced by its current offset.
static inline double si_surface(const SICore* c, double x, double y) {
    double off[3];
    si_tissue(c, off);
    return sa_surface(&c->a, x - off[0], y - off[1]) + off[2];
}

static inline void si_rotate_vessels(SICore* c) {
    double cx = 0, cy = c->a.dome[1], cs = cos(c->theta), sn = sin(c->theta);
    for (int i = 0; i < c->a.nvessel; i++) {
        const double* b = c->base_vessels[i];
        double* v = c->a.vessels[i];
        for (int k = 0; k < 2; k++) {
            double x = b[2 * k] - cx, y = b[2 * k + 1] - cy;
            v[2 * k] = cx + cs * x - sn * y;
            v[2 * k + 1] = cy + sn * x + cs * y;
        }
        v[4] = b[4];
    }
}

static inline int si_valid_site(SICore* c, double x, double y) {
    double vx, vy;
    if (sa_vessel(&c->a, x, y, &vx, &vy) < SI_SITE_VESSEL) return 0;
    for (int k = 0; k < c->nplaced; k++) {
        if (hypot(x - c->placed[k][0], y - c->placed[k][1]) < SI_SITE_THREAD) return 0;
    }
    return 1;
}

static inline void si_sample_site(SICore* c, double* x, double* y) {
    for (int tries = 0; tries < 1000; tries++) {
        double r = sqrt(sa_uniform(&c->a.rng, 0, 1)), phi = sa_uniform(&c->a.rng, 0, 2 * M_PI);
        *x = SI_SITE_SCALE * c->a.dome[3] * r * cos(phi);
        *y = c->a.dome[1] + SI_SITE_SCALE * c->a.dome[4] * r * sin(phi);
        if (si_valid_site(c, *x, *y)) return;
    }
}

static inline void si_tip_ready(const SICore* c, double* p) {
    // Needle point with the insertion slide at the ready position (the tube rides the Z stage).
    const double* q = c->a.d->qpos;
    p[0] = c->a.tip0[0] + q[c->a.qadr[0]];
    p[1] = c->a.tip0[1] + q[c->a.qadr[1]];
    p[2] = c->a.tip0[2] + q[c->a.qadr[2]] - SI_Q_READY;
}

// ---------------------------------------------------------------- setup

static inline int si_init(SICore* c, const char* path, uint64_t seed) {
    memset(c, 0, sizeof(*c));
    sa_init(&c->a, path, seed);
    memcpy(c->base_vessels, c->a.vessels, sizeof(c->base_vessels));
    // The needle enters the tissue: it does not collide with the phantom (the force model acts instead).
    // The model is shared by all worlds and read-only for stepping; this one-time change is the same for all.
    mjModel* m = (mjModel*)c->a.m;
    int needle = mj_name2id(m, mjOBJ_GEOM, "needle");
    if (needle >= 0 && (m->geom_contype[needle] || m->geom_conaffinity[needle])) {
        m->geom_contype[needle] = m->geom_conaffinity[needle] = 0;
    }
    if (needle >= 0) c->a.robot_geom[needle] = 0;
    c->needle_body = mj_name2id(m, mjOBJ_BODY, "insertion");
    return 0;
}

static inline void si_set_disturbance(SICore* c, double level_max, int mode) { sa_set_disturbance(&c->a, level_max, mode); }

static inline void si_draw_thread(SICore* c) {
    // Offset when the needle reaches the end, and the jump at release: the full simulation's values.
    double r = sa_uniform(&c->a.rng, 0, 1) < SI_DELTA_UNMOVED_P ? SI_DELTA_UNMOVED : SI_DELTA_SETTLED;
    double phi = sa_uniform(&c->a.rng, 0, 2 * M_PI);
    c->delta[0] = r * cos(phi);
    c->delta[1] = r * sin(phi);
    double j = SI_JUMPS[sa_next(&c->a.rng) % SI_JUMP_N], psi = sa_uniform(&c->a.rng, 0, 2 * M_PI);
    c->jump[0] = j * cos(psi);
    c->jump[1] = j * sin(psi);
}

static inline void si_observe(SICore* c, float* obs);

static inline double si_hover_distance(const SICore* c) {
    // True distance from the needle point to the hover pose that puts the thread end over the target.
    const double* tip = c->a.d->site_xpos + 3 * c->a.tip_site;
    double gx = c->a.goal[0] - c->delta[0], gy = c->a.goal[1] - c->delta[1];
    double gz = si_surface(c, c->a.goal[0], c->a.goal[1]) + SI_READY;
    double dx = tip[0] - gx, dy = tip[1] - gy, dz = tip[2] - gz;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

static inline void si_reset(SICore* c, float* obs) {
    const mjModel* m = c->a.m;
    mjData* d = c->a.d;
    mj_resetData(m, d);
    c->theta = sa_uniform(&c->a.rng, 0, 2 * M_PI);
    si_rotate_vessels(c);
    c->nplaced = 0;
    int n = (int)(sa_next(&c->a.rng) % (SI_MAX_PLACED + 1));
    for (int k = 0; k < n; k++) {
        double x, y;
        si_sample_site(c, &x, &y);
        c->placed[c->nplaced][0] = x;
        c->placed[c->nplaced][1] = y;
        c->nplaced++;
    }
    double sx, sy;
    si_sample_site(c, &sx, &sy);
    c->site[0] = sx;
    c->site[1] = sy;
    c->site[2] = sa_surface(&c->a, sx, sy) + SI_MARK;
    // Start: tool at rest somewhere over the field, needle point 4-10 mm above the dome.
    double x = sa_uniform(&c->a.rng, -0.04, 0.04), y = sa_uniform(&c->a.rng, -0.03, 0.03);
    double wx = c->a.tip0[0] + x, wy = c->a.tip0[1] + y;
    double tip_z = sa_surface(&c->a, wx, wy) + sa_uniform(&c->a.rng, 0.004, 0.010);
    double q[SA_NJ] = {x, y, tip_z - c->a.tip0[2] + SI_Q_READY, SI_Q_READY, SA_CARRY};
    for (int j = 0; j < SA_NJ; j++) {
        d->qpos[c->a.qadr[j]] = q[j];
        c->a.q_ref[j] = q[j];
        c->a.v_ref[j] = c->a.v_cmd[j] = c->a.integral[j] = 0;
    }
    mj_forward(m, d);
    for (int i = 0; i < 3; i++) c->a.goal0[i] = c->site[i];
    sa_draw_disturbance(&c->a);
    sa_update_goal(&c->a, 0.0);
    si_draw_thread(c);
    c->phase = SI_APPROACH;
    c->phase_t = 0;
    c->bonded = c->missed = c->dragged = c->thread_touch = c->punctured = c->entered = c->triggered = 0;
    c->tick = c->vessel_steps = 0;
    c->a.hist_count = 0;
    c->episode_return = 0;
    c->placement = c->depth = 0;
    memset(c->prev_action, 0, sizeof(c->prev_action));
    c->potential = sa_potential(si_hover_distance(c));
    si_observe(c, obs);
}

// ---------------------------------------------------------------- observation

static inline void si_observe(SICore* c, float* obs) {
    double tip[3], vel[3], goal[3];
    sa_measure(&c->a, tip, vel);  // delayed and noisy needle point
    double s = c->a.dist.sensing, lvl = c->a.dist.level;
    for (int a = 0; a < 3; a++) {
        goal[a] = c->a.goal[a];
        if (lvl > 0) goal[a] += c->a.dist.goal_bias[a] + c->a.dist.goal_drift[a] + SA_GOAL_NOISE * s * sa_normal(&c->a.drng);
    }
    for (int a = 0; a < 2; a++) {
        c->delta_meas[a] = c->delta[a] + SI_END_NOISE * (lvl > 0 ? s : 0.5) * sa_normal(&c->a.drng);
    }
    memcpy(c->a.dbg_meas_tip, tip, sizeof(tip));
    memcpy(c->a.dbg_meas_goal, goal, sizeof(goal));
    double hover[3] = {goal[0] - c->delta_meas[0], goal[1] - c->delta_meas[1], goal[2] - SI_MARK + SI_READY};
    for (int i = 0; i < 3; i++) {
        double e = tip[i] - hover[i];
        obs[i] = (float)sa_clip(e / 0.005, -5, 5);
        obs[3 + i] = (float)sa_clip(vel[i] / SA_VMAX_XY, -2, 2);
        obs[10 + i] = (float)sa_clip(e / 1e-4, -1, 1);
    }
    for (int i = 0; i < SI_ACT; i++) obs[6 + i] = c->prev_action[i];
    obs[13] = (float)sa_clip(c->delta_meas[0] / 50e-6, -2, 2);
    obs[14] = (float)sa_clip(c->delta_meas[1] / 50e-6, -2, 2);
    double vx, vy;
    sa_vessel(&c->a, tip[0], tip[1], &vx, &vy);
    obs[15] = (float)sa_clip(vx / 0.005, -1, 1);
    obs[16] = (float)sa_clip(vy / 0.005, -1, 1);
    obs[17] = (float)sa_clip((tip[2] - sa_surface(&c->a, tip[0], tip[1])) / 0.01, -1, 2);
    // The two nearest placed threads (offsets from the needle point), or far away when there are none.
    double best[2][2] = {{1, 1}, {1, 1}}, dist[2] = {1e9, 1e9};
    for (int k = 0; k < c->nplaced; k++) {
        double dx = c->placed[k][0] - tip[0], dy = c->placed[k][1] - tip[1], r = hypot(dx, dy);
        int slot = r < dist[0] ? 0 : (r < dist[1] ? 1 : -1);
        if (slot < 0) continue;
        if (slot == 0) { dist[1] = dist[0]; best[1][0] = best[0][0]; best[1][1] = best[0][1]; }
        dist[slot] = r;
        best[slot][0] = sa_clip(dx / 0.005, -1, 1);
        best[slot][1] = sa_clip(dy / 0.005, -1, 1);
    }
    for (int k = 0; k < 2; k++) { obs[18 + 2 * k] = (float)best[k][0]; obs[19 + 2 * k] = (float)best[k][1]; }
    obs[22] = c->phase == SI_APPROACH ? 0.0f : 1.0f;
    obs[23] = (float)(1.0 - (double)c->tick / SI_HORIZON);
}

// ---------------------------------------------------------------- physics

static inline double si_mj(double s, double* v, double* a, double T) {
    s = sa_clip(s, 0, 1);
    *v = (30 * s * s - 60 * s * s * s + 30 * s * s * s * s) / T;
    *a = (60 * s - 180 * s * s + 120 * s * s * s) / (T * T);
    return 10 * s * s * s - 15 * s * s * s * s + 6 * s * s * s * s * s;
}

// Needle-tissue force on the insertion body (tissue.py's model), rebuilt every physics step.
static inline void si_tissue_force(SICore* c) {
    mjData* d = c->a.d;
    double* x = d->xfrc_applied + 6 * c->needle_body;
    memset(x, 0, 6 * sizeof(double));
    const double* tip = d->site_xpos + 3 * c->a.tip_site;
    double off[3];
    si_tissue(c, off);
    double depth = si_surface(c, tip[0], tip[1]) - tip[2];
    double vz = d->qvel[c->a.dof[2]] - d->qvel[c->a.dof[3]];
    double vx = d->qvel[c->a.dof[0]], vy = d->qvel[c->a.dof[1]];
    if (depth <= 0) {
        if (c->entered && !c->punctured) c->entered = 0;  // dimple released without puncture
        return;
    }
    if (!c->entered) {
        c->entered = 1;
        c->entry[0] = tip[0] - off[0];
        c->entry[1] = tip[1] - off[1];
    }
    if (!c->punctured && depth >= SI_D_PUNCTURE) c->punctured = 1;
    double axial;
    if (!c->punctured) axial = SI_K_DIMPLE * depth;
    else axial = (vz < 0 ? SI_F_CUT : 0) * tanh(-vz / SI_V_SMOOTH) + SI_F_SHAFT * depth * tanh(-vz / SI_V_SMOOTH);
    double f[3] = {-SI_K_LATERAL * (tip[0] - off[0] - c->entry[0]) - SI_C_LATERAL * vx,
                   -SI_K_LATERAL * (tip[1] - off[1] - c->entry[1]) - SI_C_LATERAL * vy, axial};
    const double* com = d->xipos + 3 * c->needle_body;
    double r[3] = {tip[0] - com[0], tip[1] - com[1], tip[2] - com[2]};
    x[0] = f[0]; x[1] = f[1]; x[2] = f[2];
    x[3] = r[1] * f[2] - r[2] * f[1];
    x[4] = r[2] * f[0] - r[0] * f[2];
    x[5] = r[0] * f[1] - r[1] * f[0];
}

// Thread end and the placed-thread check, every physics step.
static inline void si_thread(SICore* c) {
    const double* tip = c->a.d->site_xpos + 3 * c->a.tip_site;
    double ready[3], off[3];
    si_tip_ready(c, ready);
    si_tissue(c, off);
    if (!c->bonded && c->phase <= SI_STROKE) {
        // Waiting in the tube, which rides the Z stage: 1 mm below the ready needle point, offset sideways.
        c->end[0] = ready[0] + c->delta[0];
        c->end[1] = ready[1] + c->delta[1];
        c->end[2] = ready[2] - SI_END_BELOW;
        if (c->end[2] < si_surface(c, c->end[0], c->end[1])) c->dragged = 1;
        if (c->phase == SI_STROKE && tip[2] <= c->end[2]) {
            if (hypot(c->delta[0], c->delta[1]) <= SI_STICK) c->bonded = 1;
            else c->missed = 1;
        }
    }
    if (c->bonded && c->phase <= SI_SETTLE) {
        c->end[0] = tip[0] + c->delta[0];
        c->end[1] = tip[1] + c->delta[1];
        c->end[2] = tip[2];
    } else if (c->phase >= SI_RETRACT) {
        for (int k = 0; k < 3; k++) c->end[k] = c->end_tissue[k] + off[k];
    }
    // Needle and thread tube against the threads already standing in the tissue.
    double n0[3] = {tip[0], tip[1], tip[2]}, n1[3] = {tip[0], tip[1], tip[2] + SI_NEEDLE_L};
    double e0[3] = {ready[0] + c->delta[0] + SI_TUBE_EXIT * SI_TUBE_SIN, ready[1] + c->delta[1],
                    ready[2] - SI_END_BELOW + SI_TUBE_EXIT * SI_TUBE_COS};
    double e1[3] = {e0[0] + SI_TUBE_L * SI_TUBE_SIN, e0[1], e0[2] + SI_TUBE_L * SI_TUBE_COS};
    for (int k = 0; k < c->nplaced; k++) {
        double base = si_surface(c, c->placed[k][0] + off[0], c->placed[k][1] + off[1]);
        double t0[3] = {c->placed[k][0] + off[0], c->placed[k][1] + off[1], base};
        double t1[3] = {t0[0], t0[1], base + SI_THREAD_H};
        if (si_seg_dist(n0, n1, t0, t1) < SI_NEEDLE_R + SI_THREAD_R ||
            si_seg_dist(e0, e1, t0, t1) < SI_TUBE_R + SI_THREAD_R) c->thread_touch = 1;
    }
}

// ---------------------------------------------------------------- step

static inline void si_command(SICore* c, const float* action) {
    double a[3];
    for (int i = 0; i < 3; i++) {
        a[i] = sa_clip(action[i], -1, 1);
        a[i] = a[i] * a[i] * a[i];
    }
    // X, Y and the Z stage during the approach; the insertion slide is reserved for the programmed stroke,
    // and the stages hold still from its start (no sideways motion with the needle in the tissue).
    double v[SA_NJ] = {0};
    if (c->phase == SI_APPROACH) {
        v[0] = a[0] * SA_VMAX_XY;
        v[1] = a[1] * SA_VMAX_XY;
        v[2] = a[2] * SA_VMAX_Z;
    }
    memcpy(c->a.v_cmd, v, sizeof(v));
}

static inline int si_step(SICore* c, const float* action, float* obs, float* reward, SIEpisode* episode) {
    const mjModel* m = c->a.m;
    mjData* d = c->a.d;
    double dt = m->opt.timestep, a_ref[SA_NJ];
    if (c->phase == SI_APPROACH && action[3] > 0.5f) {
        // Stroke: needle point to 2 mm below the surface as measured at the target.
        const double* tip = d->site_xpos + 3 * c->a.tip_site;
        double surface_meas = c->a.dbg_meas_goal[2] - SI_MARK;
        c->phase = SI_STROKE;
        c->phase_t = 0;
        c->triggered = 1;
        c->stroke_from = c->a.q_ref[3];
        c->stroke_len = fmax(tip[2] - (surface_meas - SI_DEPTH), SI_END_BELOW + SI_PICK_PAST);
        for (int j = 0; j < 3; j++) c->a.v_ref[j] = 0;  // stage references held where they are
    }
    si_command(c, action);
    int contact = 0;
    for (int s = 0; s < SA_REPEAT && c->phase != SI_DONE; s++) {
        mj_step1(m, d);
        sa_record(&c->a);
        for (int j = 0; j < 3; j++) {
            double next = c->a.v_ref[j] + (c->a.v_cmd[j] - c->a.v_ref[j]) * dt / SA_FILTER_S;
            double q = c->a.q_ref[j] + next * dt;
            double qmin = c->a.lo[j] + 2e-4, qmax = c->a.hi[j] - 2e-4;
            if (q < qmin || q > qmax) { q = sa_clip(q, qmin, qmax); next = 0; }
            a_ref[j] = (next - c->a.v_ref[j]) / dt;
            c->a.v_ref[j] = next;
            c->a.q_ref[j] = q;
        }
        // Insertion slide: hold, stroke, settle, retract (programmed motion, as a stage controller runs it).
        double v = 0, acc = 0, q3 = c->a.q_ref[3];
        if (c->phase == SI_STROKE) {
            double pick = SI_END_BELOW + SI_PICK_PAST, t = c->phase_t;
            if (t < SI_PICK_S) {
                q3 = c->stroke_from + pick * si_mj(t / SI_PICK_S, &v, &acc, SI_PICK_S);
                v *= pick; acc *= pick;
            } else if (t < SI_PICK_S + SI_PICK_HOLD_S) {
                q3 = c->stroke_from + pick;
            } else {
                double rest = c->stroke_len - pick;
                q3 = c->stroke_from + pick + rest * si_mj((t - SI_PICK_S - SI_PICK_HOLD_S) / SI_INSERT_S, &v, &acc, SI_INSERT_S);
                v *= rest; acc *= rest;
            }
        } else if (c->phase == SI_RETRACT && c->phase_t >= SI_RELEASE_S) {
            double back = SI_Q_READY - c->retract_from;
            q3 = c->retract_from + back * si_mj((c->phase_t - SI_RELEASE_S) / SI_RETRACT_S, &v, &acc, SI_RETRACT_S);
            v *= back; acc *= back;
        }
        a_ref[3] = acc; c->a.v_ref[3] = v; c->a.q_ref[3] = q3;
        a_ref[4] = 0; c->a.v_ref[4] = 0;
        sa_servo(&c->a, a_ref);
        sa_disturb(&c->a);
        si_tissue_force(c);
        contact |= sa_contact(&c->a);
        mj_step2(m, d);
        mj_kinematics(m, d);
        sa_update_goal(&c->a, d->time);
        si_thread(c);
        c->phase_t += dt;
        if (c->phase == SI_STROKE && c->phase_t >= SI_STROKE_S) { c->phase = SI_SETTLE; c->phase_t = 0; }
        else if (c->phase == SI_SETTLE && c->phase_t >= SI_SETTLE_S) {
            // Release: the tissue holds the end where it is, plus the end's sideways jump.
            double off[3];
            si_tissue(c, off);
            c->end_tissue[0] = c->end[0] + c->jump[0] - off[0];
            c->end_tissue[1] = c->end[1] + c->jump[1] - off[1];
            c->end_tissue[2] = c->end[2] - off[2];
            c->depth = si_surface(c, c->end[0], c->end[1]) - c->end[2];
            c->retract_from = c->a.q_ref[3];
            c->phase = SI_RETRACT; c->phase_t = 0;
        } else if (c->phase == SI_RETRACT && c->phase_t >= SI_RELEASE_S + SI_RETRACT_S) { c->phase = SI_DONE; }
        if (c->missed || c->dragged || c->thread_touch) break;
    }
    c->tick++;
    if (c->a.dist.level > 0) {
        double step = SA_REPEAT * dt;
        for (int a = 0; a < 3; a++) c->a.dist.goal_drift[a] += SA_GOAL_DRIFT * c->a.dist.sensing * sqrt(step) * sa_normal(&c->a.drng);
    }
    double r = 0;
    if (c->phase == SI_APPROACH) {
        double potential = sa_potential(si_hover_distance(c));
        r += potential - c->potential;
        c->potential = potential;
    }
    double smooth = 0;
    for (int i = 0; i < SI_ACT; i++) {
        double applied = sa_clip(action[i], -1, 1), da = applied - c->prev_action[i];
        if (i < 3 && c->phase == SI_APPROACH) smooth += da * da;
        c->prev_action[i] = (float)applied;
    }
    r -= 0.002 * smooth;
    const double* tip = d->site_xpos + 3 * c->a.tip_site;
    double vx, vy, clearance = sa_vessel(&c->a, tip[0], tip[1], &vx, &vy);
    if (c->phase == SI_APPROACH && tip[2] - si_surface(c, tip[0], tip[1]) < SA_VESSEL_HEIGHT && clearance < SA_VESSEL_MARGIN) {
        r -= 0.05;
        c->vessel_steps++;
    }
    int finite = 1;
    for (int i = 0; i < m->nq; i++) finite &= isfinite(d->qpos[i]);
    int done = 0;
    float success = 0, collision = 0, timeout = 0;
    if (contact || !finite || c->thread_touch || c->dragged || c->missed) {
        r -= 2;
        done = 1;
        collision = contact || !finite;
    } else if (c->phase == SI_DONE) {
        double off[3];
        si_tissue(c, off);
        c->placement = hypot(c->end_tissue[0] - c->site[0], c->end_tissue[1] - c->site[1]);
        r += 2 * exp(-c->placement / SI_TOL);
        success = c->placement < SI_TOL;
        done = 1;
    } else if (c->tick >= SI_HORIZON) {
        done = 1;
        timeout = 1;
    }
    *reward = (float)r;
    c->episode_return += r;
    if (done && episode) {
        episode->success = success;
        episode->collision = collision;
        episode->thread_touch = (float)c->thread_touch;
        episode->dragged = (float)c->dragged;
        episode->missed = (float)c->missed;
        episode->timeout = timeout;
        episode->placement_um = (float)(c->phase == SI_DONE ? c->placement * 1e6 : 0);
        episode->depth_um = (float)(c->depth * 1e6);
        episode->vessel_steps = (float)c->vessel_steps;
        episode->episode_return = (float)c->episode_return;
        episode->episode_length = (float)c->tick;
        episode->level = (float)c->a.dist.level;
        episode->nplaced = (float)c->nplaced;
    }
    si_observe(c, obs);
    return done;
}

// Scripted yardsticks on the same observation. compensate = 0: aim the needle at the target (the MuJoCo
// cycle's yardstick); 1: aim so the thread end lands on the target, from the measured offset. Approach at
// least 3 mm above the dome until within 0.5 mm laterally, then hover and start the stroke once settled.
// Programmed control, used as yardsticks for learning, not as deployed components.
static inline void si_scripted(const float* obs, float* action, int compensate) {
    const double gain = 6.0;
    double vmax[3] = {SA_VMAX_XY, SA_VMAX_XY, SA_VMAX_Z}, e[3], v[3];
    for (int i = 0; i < 3; i++) e[i] = fabs(obs[10 + i]) < 1 ? obs[10 + i] * 1e-4 : obs[i] * 0.005;
    if (!compensate) { e[0] -= obs[13] * 50e-6; e[1] -= obs[14] * 50e-6; }
    for (int i = 0; i < 3; i++) v[i] = -gain * e[i];
    double lateral = sqrt(e[0] * e[0] + e[1] * e[1]), height = obs[17] * 0.01;
    if (lateral > 5e-4 && gain * (SA_VESSEL_HEIGHT - height) > v[2]) v[2] = gain * (SA_VESSEL_HEIGHT - height);
    for (int i = 0; i < 3; i++) action[i] = (float)cbrt(sa_clip(v[i], -vmax[i], vmax[i]) / vmax[i]);
    double speed = sqrt(obs[3] * obs[3] + obs[4] * obs[4] + obs[5] * obs[5]) * SA_VMAX_XY;
    action[3] = (lateral < 5e-6 && fabs(e[2]) < 2e-5 && speed < 2e-4) ? 1.0f : 0.0f;
}

static inline void si_free(SICore* c) { sa_free(&c->a); }
