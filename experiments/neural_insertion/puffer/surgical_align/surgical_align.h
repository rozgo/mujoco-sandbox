// PufferLib 5.0 adapter for the surgical alignment task (pinned revision
// 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2). Physics, servo, observations and
// rewards live in surgical_core.h, shared with the local tests. Continuous
// actions: three bounded needle-tip velocity commands at 50 Hz.
#include <stdlib.h>
#include <math.h>
typedef float obs_t;
#include "pufferenv.h"
#include "surgical_core.h"

#define OBS_SIZE SA_OBS
#define NUM_ATNS SA_ACT
#define ACT_SIZES {1, 1, 1}

struct Log {
    float perf;              // success rate
    float score;             // episode return
    float episode_return;
    float episode_length;
    float collision;
    float timeout;
    float vessel_steps;
    float final_lateral_um;
    float final_vertical_um;
    float n;                 // required last field
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;               // trainer bookkeeping (native backend)
    int boundary_reached;  // trainer bookkeeping (native backend)
    int num_agents;
    unsigned int rng;
    SACore core;
};

static const char* sa_model_path(void) {
    const char* path = getenv("SURGICAL_ALIGN_XML");
    return path ? path : "resources/surgical_align/surgical_align.xml";
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    uint64_t seed = (uint64_t)dict_get(kwargs, "seed") * 1000003ULL + env->rng;
    sa_init(&env->core, sa_model_path(), seed);
}

void puf_reset(Env* env) {
    sa_reset(&env->core, env->agents[0].observations);
}

void puf_step(Env* env) {
    SAEpisode e;
    float reward;
    int done = sa_step(&env->core, env->agents[0].actions, env->agents[0].observations, &reward, &e);
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = (float)done;
    if (done) {
        env->log.perf += e.success;
        env->log.score += e.episode_return;
        env->log.episode_return += e.episode_return;
        env->log.episode_length += e.episode_length;
        env->log.collision += e.collision;
        env->log.timeout += e.timeout;
        env->log.vessel_steps += e.vessel_steps;
        env->log.final_lateral_um += e.final_lateral_um;
        env->log.final_vertical_um += e.final_vertical_um;
        env->log.n += 1;
        sa_reset(&env->core, env->agents[0].observations);
    }
}

void puf_render(Env* env) {
    // Rendering is done offline from recorded states with the MuJoCo renderer.
}

void puf_close(Env* env) {
    sa_free(&env->core);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "collision", log->collision);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "vessel_steps", log->vessel_steps);
    dict_set(out, "final_lateral_um", log->final_lateral_um);
    dict_set(out, "final_vertical_um", log->final_vertical_um);
    dict_set(out, "n", log->n);
}
