// PufferLib 5.0 adapter for the thread insertion task (pinned revision
// 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2). Physics, servo, disturbances, observations and rewards
// live in insert_core.h (built on surgical_core.h), shared with the local tests. Continuous actions:
// three bounded needle-point velocity commands and a stroke trigger, at 50 Hz.
#include <stdlib.h>
#include <math.h>
typedef float obs_t;
#include "pufferenv.h"
#include "insert_core.h"

#define OBS_SIZE SI_OBS
#define NUM_ATNS SI_ACT
#define ACT_SIZES {1, 1, 1, 1}

struct Log {
    float perf;              // success rate: thread end within 10 µm of the target
    float score;             // episode return
    float episode_return;
    float episode_length;
    float placed;            // stroke completed and the thread released (any placement)
    float placement_um;      // summed over placed episodes only (divide by `placed` for the mean)
    float collision;
    float thread_touch;
    float dragged;
    float missed;
    float timeout;
    float vessel_steps;
    float level;
    float n;                 // required last field
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;
    SICore core;
};

static const char* si_model_path(void) {
    const char* path = getenv("SURGICAL_INSERT_XML");
    return path ? path : "resources/surgical_insert/surgical_align.xml";
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    uint64_t seed = (uint64_t)dict_get(kwargs, "seed") * 1000003ULL + env->rng;
    si_init(&env->core, si_model_path(), seed);
    si_set_disturbance(&env->core, dict_get(kwargs, "disturbance"), (int)dict_get(kwargs, "disturbance_mode"));
}

void puf_reset(Env* env) {
    si_reset(&env->core, env->agents[0].observations);
}

void puf_step(Env* env) {
    SIEpisode e;
    float reward;
    int done = si_step(&env->core, env->agents[0].actions, env->agents[0].observations, &reward, &e);
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = (float)done;
    if (done) {
        float placed = e.collision + e.thread_touch + e.dragged + e.missed + e.timeout == 0 ? 1.0f : 0.0f;
        env->log.perf += e.success;
        env->log.score += e.episode_return;
        env->log.episode_return += e.episode_return;
        env->log.episode_length += e.episode_length;
        env->log.placed += placed;
        env->log.placement_um += placed * e.placement_um;
        env->log.collision += e.collision;
        env->log.thread_touch += e.thread_touch;
        env->log.dragged += e.dragged;
        env->log.missed += e.missed;
        env->log.timeout += e.timeout;
        env->log.vessel_steps += e.vessel_steps;
        env->log.level += e.level;
        env->log.n += 1;
        si_reset(&env->core, env->agents[0].observations);
    }
}

void puf_render(Env* env) {
    // Rendering is done offline from recorded states.
}

void puf_close(Env* env) {
    si_free(&env->core);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "placed", log->placed);
    dict_set(out, "placement_um", log->placement_um);
    dict_set(out, "collision", log->collision);
    dict_set(out, "thread_touch", log->thread_touch);
    dict_set(out, "dragged", log->dragged);
    dict_set(out, "missed", log->missed);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "vessel_steps", log->vessel_steps);
    dict_set(out, "level", log->level);
    dict_set(out, "n", log->n);
}
