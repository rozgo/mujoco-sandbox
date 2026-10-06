// Shared library around PufferLib's CPU network (src/puffercpu.c) so Python
// tools can run a trained alignment checkpoint: deterministic Gaussian mean.
#include "puffercpu.c"

typedef struct {
    PufferNet* net;
    Weights* weights;
} PolicyHandle;

void* policy_create(const char* path, int obs, int hidden, int layers, int actions) {
    Weights* weights = load_weights(path);
    if (!weights) {
        return NULL;
    }
    int act_sizes[16];
    for (int i = 0; i < actions; i++) {
        act_sizes[i] = 1;
    }
    PolicyHandle* h = (PolicyHandle*)calloc(1, sizeof(PolicyHandle));
    h->weights = weights;
    h->net = make_puffernet(weights, 1, obs, hidden, layers, act_sizes, actions);
    if (weights->idx > weights->size || weights->size - 7 - weights->idx > 7) {
        free(h);
        return NULL;
    }
    return h;
}

// terminal = 1 zeroes the recurrent state (start of an episode).
void policy_act(void* p, float* obs, float terminal, float* action) {
    PolicyHandle* h = (PolicyHandle*)p;
    forward_puffernet(h->net, obs, action, NULL, &terminal);
}

// The learned, state-independent action noise (log standard deviation per action).
void policy_log_std(void* p, float* out) {
    PolicyHandle* h = (PolicyHandle*)p;
    for (int i = 0; i < h->net->num_actions; i++) out[i] = h->net->log_std[i];
}
