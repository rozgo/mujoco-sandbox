// Batched, unchanged MuJoCo stepping with per-substep contact/work telemetry.
// implicit/Euler: mj_step1 + mj_step2 (identical to the repository kernel).
// RK4: the same sequence mj_step performs (checks, mj_forward, mj_RungeKutta),
// with telemetry taken at the stage-0 evaluation, since mj_step2 ignores RK4.
// support < 0 inspects every contact (penetration/force telemetry only).
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" void contact_chunk(const mjModel* m, mjData* d, int steps,
                              int support, int free_body, double* out) {
  std::fill(out, out+15, 0.);
  out[12] = -1;
  out[13] = -1;
  std::vector<mjtNum> previous(m->nv);
  const bool rk = m->opt.integrator == mjINT_RK4;
  for (int step=0; step<steps; ++step) {
    if (rk) {
      mj_checkPos(m, d);
      mj_checkVel(m, d);
      mj_forward(m, d);
      mj_checkAcc(m, d);
    } else {
      mj_step1(m, d);
    }
    mju_zero(d->qfrc_applied, m->nv);
    mju_copy(previous.data(), d->qvel, m->nv);
    const double stamp = d->time;
    // Contacts and constraint forces below belong to the pre-integration state.
    std::vector<mjtNum> qfrc(m->nv);
    if (rk) {
      mju_copy(qfrc.data(), d->qfrc_constraint, m->nv);
      bool touching = false;
      for (int i=0; i<d->ncon; ++i) {
        const mjContact& c = d->contact[i];
        if (support >= 0 && c.geom[0] != support && c.geom[1] != support) continue;
        out[1] = std::max(out[1], -c.dist);
        mjtNum local[6], force[3], torque[3], moment[3];
        mj_contactForce(m, d, i, local);
        out[2] = std::max(out[2], std::abs(local[0]));
        touching |= local[0] > 0;
        mju_mulMatTVec(force, c.frame, local, 3, 3);
        mju_mulMatTVec(torque, c.frame, local+3, 3, 3);
        double sign = c.geom[0] == support ? 1 : -1;
        mju_scl3(force, force, sign);
        mju_scl3(torque, torque, sign);
        mju_cross(moment, c.pos, force);
        for (int k=0; k<3; ++k) {
          out[4+k] += m->opt.timestep*force[k];
          out[7+k] += m->opt.timestep*(moment[k]+torque[k]);
        }
      }
      if (touching) {
        out[10] += 1;
        if (out[12] < 0) out[12] = stamp;
        out[13] = stamp;
      }
      for (int i=0; i<mjNISLAND; ++i) out[11] = std::max(out[11], double(d->solver_niter[i]));
      mj_RungeKutta(m, d, 4);
      for (int k=0; k<m->nv; ++k)
        out[3] += m->opt.timestep*qfrc[k]*(previous[k]+d->qvel[k])/2;
    } else {
      mj_step2(m, d);
      for (int k=0; k<m->nv; ++k)
        out[3] += m->opt.timestep*d->qfrc_constraint[k]*(previous[k]+d->qvel[k])/2;
      bool touching = false;
      for (int i=0; i<d->ncon; ++i) {
        const mjContact& c = d->contact[i];
        if (support >= 0 && c.geom[0] != support && c.geom[1] != support) continue;
        out[1] = std::max(out[1], -c.dist);
        mjtNum local[6], force[3], torque[3], moment[3];
        mj_contactForce(m, d, i, local);
        out[2] = std::max(out[2], std::abs(local[0]));
        touching |= local[0] > 0;
        mju_mulMatTVec(force, c.frame, local, 3, 3);
        mju_mulMatTVec(torque, c.frame, local+3, 3, 3);
        double sign = c.geom[0] == support ? 1 : -1;
        mju_scl3(force, force, sign);
        mju_scl3(torque, torque, sign);
        mju_cross(moment, c.pos, force);
        for (int k=0; k<3; ++k) {
          out[4+k] += m->opt.timestep*force[k];
          out[7+k] += m->opt.timestep*(moment[k]+torque[k]);
        }
      }
      if (touching) {
        out[10] += 1;
        if (out[12] < 0) out[12] = stamp;
        out[13] = stamp;
      }
      for (int i=0; i<mjNISLAND; ++i) out[11] = std::max(out[11], double(d->solver_niter[i]));
    }
    if (free_body >= 0) {
      mjtNum gravity_force[3], moment[3];
      mju_scl3(gravity_force, m->opt.gravity, m->body_mass[free_body]);
      mju_cross(moment, d->xipos+3*free_body, gravity_force);
      for (int k=0; k<3; ++k) out[7+k] += m->opt.timestep*moment[k];
    }
    out[0] += 1;
    for (int i=0; i<mjNWARNING; ++i) out[14] = std::max(out[14], double(d->warning[i].number != 0));
    for (int i=0; i<m->nq; ++i) if (!std::isfinite(d->qpos[i])) out[14] = 1;
    for (int i=0; i<m->nv; ++i) if (!std::isfinite(d->qvel[i])) out[14] = 1;
    if (out[14]) break;
  }
}
