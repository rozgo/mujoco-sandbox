// Evaluate a PufferLib 5.0 checkpoint on predetermined alignment seeds.
// Uses Puffer's own CPU network code (src/puffercpu.c, deterministic Gaussian
// mean) and the shared alignment core, so evaluation runs the trained policy on
// exactly the physics and observations used in training. One fresh world per
// seed, matching align_env.AlignEnv(seed).
//
// usage: align_eval MODEL.xml WEIGHTS.bin HIDDEN LAYERS FIRST_SEED COUNT LEVEL OUT_DIR [RECORD_SEED...]
#include "puffercpu.c"
#include "surgical_core.h"

#include <sys/stat.h>

static void write_states(const char* dir, uint64_t seed, const double* rows, int n) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/states_%llu.bin", dir, (unsigned long long)seed);
    FILE* f = fopen(path, "wb");
    fwrite(rows, sizeof(double), (size_t)n * SA_ROW, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s MODEL.xml WEIGHTS.bin HIDDEN LAYERS FIRST_SEED COUNT LEVEL OUT_DIR [RECORD_SEED...]\n", argv[0]);
        return 1;
    }
    const char* xml = argv[1];
    Weights* weights = load_weights(argv[2]);
    if (!weights) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    int hidden = atoi(argv[3]), layers = atoi(argv[4]);
    uint64_t first = strtoull(argv[5], NULL, 10);
    int count = atoi(argv[6]);
    double level = atof(argv[7]);
    const char* out = argv[8];
    mkdir(out, 0755);
    int act_sizes[SA_ACT] = {1, 1, 1};
    PufferNet* net = make_puffernet(weights, 1, SA_OBS, hidden, layers, act_sizes, SA_ACT);
    if (weights->idx > weights->size || weights->size - 7 - weights->idx > 7) {
        fprintf(stderr, "weight count mismatch: read %d of %d\n", weights->idx, weights->size - 7);
        return 1;
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/episodes.json", out);
    FILE* json = fopen(path, "w");
    fprintf(json, "[\n");
    float obs[SA_OBS], action[SA_ACT], reward, terminal;
    double* rows = (double*)malloc(sizeof(double) * SA_ROW * (SA_HORIZON + 1));
    for (int k = 0; k < count; k++) {
        uint64_t seed = first + (uint64_t)k;
        int record = 0;
        for (int i = 9; i < argc; i++) {
            record |= strtoull(argv[i], NULL, 10) == seed;
        }
        SACore core;
        sa_init(&core, xml, seed);
        sa_set_disturbance(&core, level, 0);
        sa_reset(&core, obs);
        terminal = 1;  // zero the recurrent state at the start of every episode
        SAEpisode e;
        int n = 0;
        float last_norm = 0;
        for (;;) {
            if (record) {
                sa_replay_row(&core, last_norm, rows + SA_ROW * n++);
            }
            forward_puffernet(net, obs, action, NULL, &terminal);
            terminal = 0;
            last_norm = sqrtf(action[0] * action[0] + action[1] * action[1] + action[2] * action[2]);
            if (sa_step(&core, action, obs, &reward, &e)) {
                break;
            }
        }
        if (record) {
            sa_replay_row(&core, last_norm, rows + SA_ROW * n++);
            write_states(out, seed, rows, n);
        }
        fprintf(json, "  {\"seed\": %llu, \"target\": %d, \"success\": %g, \"collision\": %g, \"timeout\": %g, "
                "\"vessel_steps\": %g, \"final_lateral_um\": %.6g, \"final_vertical_um\": %.6g, "
                "\"episode_return\": %.6g, \"episode_length\": %g, \"level\": %g}%s\n",
                (unsigned long long)seed, core.target, e.success, e.collision, e.timeout, e.vessel_steps,
                e.final_lateral_um, e.final_vertical_um, e.episode_return, e.episode_length,
                e.level, k + 1 < count ? "," : "");
        sa_free(&core);
    }
    fprintf(json, "]\n");
    fclose(json);
    free(rows);
    return 0;
}
