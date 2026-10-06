// Local test/evaluation library around the shared insertion core (ctypes).
#include "insert_core.h"

void* si_create(const char* path, unsigned long long seed) {
    SICore* c = (SICore*)calloc(1, sizeof(SICore));
    si_init(c, path, seed);
    return c;
}

void si_disturbance_c(void* p, double level, int mode) { si_set_disturbance((SICore*)p, level, mode); }

void si_destroy(void* p) {
    si_free((SICore*)p);
    free(p);
}

void si_reset_c(void* p, float* obs) { si_reset((SICore*)p, obs); }

// episode[13]: see SIEpisode
int si_step_c(void* p, const float* action, float* obs, float* reward, float* episode) {
    SIEpisode e;
    int done = si_step((SICore*)p, action, obs, reward, &e);
    if (done) memcpy(episode, &e, sizeof(e));
    return done;
}

void si_scripted_c(const float* obs, float* action, int compensate) { si_scripted(obs, action, compensate); }

// Episode setup for diagnostics: site (3), delta (2), jump (2), nplaced, theta, phase, tip (3), end (3).
void si_state(void* p, double* out) {
    SICore* c = (SICore*)p;
    memcpy(out, c->site, 3 * sizeof(double));
    memcpy(out + 3, c->delta, 2 * sizeof(double));
    memcpy(out + 5, c->jump, 2 * sizeof(double));
    out[7] = c->nplaced;
    out[8] = c->theta;
    out[9] = c->phase;
    memcpy(out + 10, c->a.d->site_xpos + 3 * c->a.tip_site, 3 * sizeof(double));
    memcpy(out + 13, c->end, 3 * sizeof(double));
}
