"""Joint-only locomotion and arm control. The floating base is never driven directly."""
import math
import mujoco
import numpy as np

LEGS = [f"{s}_{p}" for s in ("left", "right") for p in ("front", "middle", "rear")]


def leg_ik(point, mount, angle):
    """Closed-form 3-DOF IK for the approved coxa/femur/tibia geometry."""
    p = np.asarray(point) - mount
    yaw = math.atan2(p[1], p[0]) - angle
    yaw = (yaw + math.pi) % (2*math.pi) - math.pi
    r, z = math.hypot(p[0], p[1]) - .22, p[2]
    a, b = math.hypot(.36,.08), math.hypot(.24,.78)
    dist = math.hypot(r,z)
    theta = math.atan2(z,r) + math.acos(np.clip((a*a+dist*dist-b*b)/(2*a*dist),-1,1))
    hip = math.atan2(.08,.36)-theta
    knee = math.atan2(-.78,.24)-math.atan2(.08,.36)+math.acos(np.clip((dist*dist-a*a-b*b)/(2*a*b),-1,1))
    return np.array([yaw,hip,knee])


class Walker:
    """Slow five-foot-support ripple gait with world-fixed stance targets."""
    def __init__(self, model, data):
        self.m = model
        self.reference = data.qpos[:3].copy()
        self.reference[2] = .79
        self.velocity = np.zeros(2)
        self.mounts, self.angles, self.actuators, self.nominal = [], [], [], []
        self.feet = np.array([data.site(n+"_toe").xpos.copy() for n in LEGS])
        for name in LEGS:
            self.mounts.append(model.body(name+"_coxa").pos.copy())
            sign = 1 if name.startswith("left") else -1
            index = ("front","middle","rear").index(name.split("_")[1])
            self.angles.append(sign * (math.pi/4 + index*math.pi/4))
            self.actuators.append([model.actuator(name+"_"+j).id for j in ("yaw","hip","knee")])
            self.nominal.append(self.feet[len(self.nominal),:2]-data.qpos[:2])
        self.nominal = np.array(self.nominal)
        self.order = [0,5,1,3,2,4]
        self.step_index = -1
        self.swing_start = self.feet[0].copy()
        self.swing_end = self.feet[0].copy()
        self.time = 0.
        self.active = False
        self.period = .36
        self.stop_index = None

    def update(self, data, goal, dt, walking=True):
        delta = np.asarray(goal)[:2] - data.qpos[:2]
        distance = np.linalg.norm(delta)
        desired_v = delta / max(distance,1e-9) * min(.12, math.sqrt(2*.12*distance)) if walking else np.zeros(2)
        change = desired_v-self.velocity
        self.velocity += change * min(1., .25*dt/max(np.linalg.norm(change),1e-9))
        self.reference[:2] += self.velocity*dt
        error = self.reference[:2]-data.qpos[:2]
        self.reference[:2] = data.qpos[:2]+error*min(1.,.035/max(np.linalg.norm(error),1e-9))
        if walking:
            self.active = True
            self.stop_index = None
        elif self.active and self.stop_index is None:
            self.stop_index = int(self.time/self.period)+6
        if self.active:
            self.time += dt
            index = int(self.time/self.period)
            phase = (self.time/self.period) % 1
            leg = self.order[index % 6]
            if index != self.step_index:
                if self.step_index >= 0:
                    previous = self.order[self.step_index%6]
                    self.feet[previous] = data.site(LEGS[previous]+"_toe").xpos.copy()
                    self.feet[previous,2] = .04
                self.step_index = index
                self.swing_start = self.feet[leg].copy()
                self.swing_end = np.r_[self.reference[:2]+self.nominal[leg]+self.velocity*.95, .04]
                if self.stop_index is not None and index >= self.stop_index:
                    self.active = False
            if self.active:
                s = phase*phase*(3-2*phase)
                self.feet[leg] = (1-s)*self.swing_start+s*self.swing_end
                self.feet[leg,2] += .085*math.sin(math.pi*phase)**2
        for i in range(6):
            q = leg_ik(self.feet[i]-self.reference,self.mounts[i],self.angles[i])
            ids = self.actuators[i]
            data.ctrl[ids] = np.clip(q,self.m.actuator_ctrlrange[ids,0],self.m.actuator_ctrlrange[ids,1])
        return distance < .015 and np.linalg.norm(self.velocity)<.025


class ArmIK:
    """Damped Jacobian IK on scratch data; applies targets only through actuators."""
    def __init__(self, model):
        self.m = model
        self.scratch = mujoco.MjData(model)
        self.jp = np.zeros((3,model.nv))
        self.jr = np.zeros((3,model.nv))
        self.ids = {}
        for side in ('left','right'):
            names = [f'{side}_arm_joint_{i}' for i in range(1,8)]
            self.ids[side] = (
                np.array([model.joint(n).qposadr[0] for n in names]),
                np.array([model.joint(n).dofadr[0] for n in names]),
                np.array([model.actuator(n).id for n in names]),
                model.site(side+'_grip_pinch').id,
            )

    def solve(self, data, side, position, rotation, seed=None, iterations=100):
        qadr,dof,act,site = self.ids[side]
        s = self.scratch
        s.qpos[:] = data.qpos
        if seed is not None:
            s.qpos[qadr] = seed
        lo,hi = self.m.actuator_ctrlrange[act].T
        for _ in range(iterations):
            mujoco.mj_kinematics(self.m,s)
            mujoco.mj_comPos(self.m,s)
            current = s.site_xmat[site].reshape(3,3)
            ep = np.asarray(position)-s.site_xpos[site]
            er = .5*sum((np.cross(current[:,i],rotation[:,i]) for i in range(3)))
            error = np.r_[ep,er*.5]
            if np.linalg.norm(ep)<.0003 and np.linalg.norm(er)<.002:
                break
            mujoco.mj_jacSite(self.m,s,self.jp,self.jr,site)
            j = np.vstack([self.jp[:,dof],self.jr[:,dof]*.5])
            dq = j.T @ np.linalg.solve(j@j.T+np.eye(6)*.00003,error)
            dq *= min(1.,.15/max(np.linalg.norm(dq),1e-9))
            s.qpos[qadr] = np.clip(s.qpos[qadr]+dq,lo+1e-5,hi-1e-5)
        return s.qpos[qadr].copy(),float(np.linalg.norm(ep)),float(np.linalg.norm(er))


# Gripper advances along world +X; its jaws close across world Y.
SIDE_GRASP = np.array([[0,0,1],[0,-1,0],[1,0,0]],dtype=float)
TOP_GRASP = np.diag([1.,-1.,-1.])
