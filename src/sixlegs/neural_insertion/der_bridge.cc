// MuJoCo 3.12 lifecycle adapter for the pinned MIT-licensed upstream DER.
// The upstream elastic-force formulas are unchanged. See der_build.py.
#include <mujoco/mujoco.h>
#include <mujoco/mjplugin.h>
#include "wire_qst.h"
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using mujoco::plugin::elasticity::WireQST;

static WireQST* rod(const mjData* d, int instance) {
  return reinterpret_cast<WireQST*>(d->plugin_data[instance]);
}

mjPLUGIN_LIB_INIT(sixlegs_der) {
  mjpPlugin p;
  mjp_defaultPlugin(&p);
  p.name = "sixlegs.der_qst";
  p.capabilityflags = mjPLUGIN_PASSIVE;
  p.needstage = mjSTAGE_POS;
  static const char* attributes[] = {"twist", "bend", "flat", "vmax", "twist_displace", "projection"};
  p.attributes = attributes;
  p.nattribute = 6;
  // Previous wrapped and accumulated twist are simulation state. Compute must
  // not advance that history when the viewer, sensors or mj_forward evaluate it.
  p.nstate = +[](const mjModel*, int) { return 2; };
  p.init = +[](const mjModel* m, mjData* d, int instance) {
    auto value = WireQST::Create(m, d, instance);
    if (!value) return -1;
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(new WireQST(std::move(*value)));
    return 0;
  };
  p.destroy = +[](mjData* d, int instance) {
    delete rod(d, instance);
    d->plugin_data[instance] = 0;
  };
  p.copy = +[](mjData* dest, const mjModel*, const mjData* src, int instance) {
    if (dest == src) return;
    delete rod(dest, instance);
    dest->plugin_data[instance] = reinterpret_cast<uintptr_t>(new WireQST(*rod(src, instance)));
  };
  p.reset = +[](const mjModel* m, mjtNum* state, void*, int instance) {
    state[1] = std::strtod(mj_getPluginConfig(m, instance, "twist_displace"), nullptr);
    state[0] = std::remainder(state[1], 2*mjPI);
  };
  p.compute = +[](const mjModel* m, mjData* d, int instance, int) {
    auto* r = rod(d, instance);
    const bool direct = std::string(mj_getPluginConfig(m, instance, "projection")) == "direct";
    std::vector<mjtNum> before;
    if (direct) before.assign(d->qfrc_passive, d->qfrc_passive+m->nv);
    const auto* s = d->plugin_state + m->plugin_stateadr[instance];
    r->p_thetan = s[0];
    r->edges[r->nv].theta = s[1];
    r->edges[0].theta = 0;
    r->Compute(m, d, instance);
    if (direct) {
      // Preserve MuJoCo damping and other contributors; replace only the
      // upstream lever projection. Project the same Cartesian elastic forces
      // through MuJoCo's Jacobians, and include material-frame end moments.
      mju_copy(d->qfrc_passive, before.data(), m->nv);
      const mjtNum zero[3] = {0, 0, 0};
      for (int i=0; i<r->nv+2; ++i) {
        int body = r->i0 + std::min(i, r->nv);
        mj_applyFT(m, d, r->nodes[i].force.data(), zero, r->nodes[i].pos.data(),
                   body, d->qfrc_passive);
      }
      double moment = r->beta_bar*(r->edges[r->nv].theta-r->edges[0].theta)/r->bigL_bar;
      Eigen::Vector3d first = moment*r->edges[0].e.normalized();
      Eigen::Vector3d last = -moment*r->edges[r->nv].e.normalized();
      mj_applyFT(m, d, zero, first.data(), r->nodes[0].pos.data(), r->i0, d->qfrc_passive);
      mj_applyFT(m, d, zero, last.data(), r->nodes[r->nv+1].pos.data(),
                 r->i0+r->nv, d->qfrc_passive);
    }
  };
  p.advance = +[](const mjModel* m, mjData* d, int instance) {
    auto* s = d->plugin_state + m->plugin_stateadr[instance];
    s[0] = rod(d, instance)->p_thetan;
    s[1] = rod(d, instance)->edges[rod(d, instance)->nv].theta;
  };
  mjp_registerPlugin(&p);
}

extern "C" void sixlegs_der_metrics(const mjModel* m, const mjData* d, double* out) {
  auto* r = rod(d, 0);
  double bending = 0;
  for (int i=1; i<=r->nv; ++i)
    bending += r->alpha_bar*r->nodes[i].kb.squaredNorm()/r->edges[i].l_bar;
  double theta = r->edges[r->nv].theta - r->edges[0].theta;
  out[0] = bending;
  out[1] = r->beta_bar*theta*theta/(2*r->bigL_bar);
  out[2] = theta;
  out[3] = r->bigL_bar;
}
