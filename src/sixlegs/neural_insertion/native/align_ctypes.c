// Local test/evaluation library around the shared alignment core (ctypes).
#include "surgical_core.h"

typedef struct {
    SACore core;
} SAHandle;

void* sa_create(const char* path, unsigned long long seed) {
    SAHandle* h = (SAHandle*)calloc(1, sizeof(SAHandle));
    sa_init(&h->core, path, seed);
    return h;
}

void sa_destroy(void* p) {
    SAHandle* h = (SAHandle*)p;
    sa_free(&h->core);
    free(h);
}

void sa_reset_c(void* p, float* obs) {
    sa_reset(&((SAHandle*)p)->core, obs);
}

// episode[8] = success, collision, timeout, vessel_steps, lateral_um, vertical_um, return, length
int sa_step_c(void* p, const float* action, float* obs, float* reward, float* episode) {
    SAEpisode e;
    int done = sa_step(&((SAHandle*)p)->core, action, obs, reward, &e);
    if (done) {
        memcpy(episode, &e, sizeof(e));
    }
    return done;
}

void sa_scripted_c(const float* obs, float* action) {
    sa_scripted(obs, action);
}

// state: tip(3) goal(3) qpos(5) q_ref(5) target tick
void sa_state(void* p, double* out) {
    SACore* c = &((SAHandle*)p)->core;
    memcpy(out, c->d->site_xpos + 3 * c->tip_site, 3 * sizeof(double));
    memcpy(out + 3, c->goal, 3 * sizeof(double));
    for (int j = 0; j < SA_NJ; j++) {
        out[6 + j] = c->d->qpos[c->qadr[j]];
        out[11 + j] = c->q_ref[j];
    }
    out[16] = c->target;
    out[17] = c->tick;
}

// Full scripted-policy episodes in C; results[8*k] as in sa_step_c.
void sa_rollout_scripted(void* p, int episodes, float* results) {
    float obs[SA_OBS], action[SA_ACT], reward;
    for (int k = 0; k < episodes; k++) {
        sa_reset_c(p, obs);
        for (;;) {
            sa_scripted(obs, action);
            if (sa_step_c(p, action, obs, &reward, results + 8 * k)) {
                break;
            }
        }
    }
}
