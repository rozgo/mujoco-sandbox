"""Deterministic, fully dynamic two-object transfer demonstration."""
from dataclasses import dataclass
import math

import mujoco
import numpy as np

from sixlegs.control import ArmIK, SIDE_GRASP, Walker


@dataclass(frozen=True)
class Phase:
    name: str
    duration: float = 0
    goal: tuple | None = None
    pose: tuple | None = None  # world pinch X, Z; Y is the object's lane
    closed: bool = False
    carry: bool = False
    destination: bool = False


PHASES = [
    Phase('approach source', goal=(-.95,0)),
    Phase('settle at source',3),
    Phase('align open grippers',3,pose=(-.03,1.08)),
    Phase('lower beside objects',2,pose=(-.03,.89)),
    Phase('reach around mug and block',2,pose=(.13,.89)),
    Phase('close both grippers',2,pose=(.13,.89),closed=True),
    Phase('lift both objects',3,pose=(.13,1.16),closed=True),
    Phase('tuck arms for carrying',3,closed=True,carry=True),
    Phase('back away from source',goal=(-2.7,0),closed=True,carry=True),
    Phase('walk around barrier',goal=(-2.7,3.6),closed=True,carry=True),
    Phase('approach destination',goal=(-.95,3.6),closed=True,carry=True),
    Phase('settle at destination',3,closed=True,carry=True),
    Phase('extend above placement marks',3,pose=(.13,1.16),closed=True,destination=True),
    Phase('lower onto destination',3,pose=(.13,.892),closed=True,destination=True),
    Phase('release both objects',2,pose=(.13,.892),destination=True),
    Phase('withdraw open grippers',2,pose=(-.03,.94),destination=True),
    Phase('raise clear of table',2,pose=(-.03,1.16),destination=True),
    Phase('verify placement',3,pose=(-.03,1.16),destination=True),
]


class TransferDemo:
    def __init__(self, model, data):
        self.m, self.d = model,data
        self.walker = Walker(model,data)
        self.ik = ArmIK(model)
        self.index = 0
        self.started = data.time
        self.seeds = {}
        self.events = []
        self.starts = self.pinch_positions()
        self.done = False
        self.success = False
        self.max_tracking_error = 0.
        self.min_carry_height = {'mug': math.inf,'block':math.inf}
        self.robot_bodies = set()
        root = model.body('chassis').id
        for i in range(1,model.nbody):
            ancestor=i
            while ancestor and ancestor!=root: ancestor=int(model.body_parentid[ancestor])
            if ancestor==root:self.robot_bodies.add(i)
        self.forbidden_contacts = []
        self.announce()

    @property
    def phase(self):
        return PHASES[self.index]

    def announce(self):
        self.events.append({'phase':self.phase.name,'time':round(self.d.time,3)})
        print(f'[{self.d.time:6.1f}s] {self.phase.name}',flush=True)

    def pinch_positions(self):
        return {s:self.d.site(s+'_grip_pinch').xpos.copy() for s in ('left','right')}

    def pad_contacts(self, side, object_name):
        faces=set()
        for c in self.d.contact:
            bodies=[self.m.body(int(self.m.geom_bodyid[g])).name for g in c.geom]
            names=[self.m.geom(int(g)).name for g in c.geom]
            if object_name in bodies:
                for name in names:
                    for face in ('left','right'):
                        if name.startswith(f'{side}_grip_{face}_pad'): faces.add(face)
        return sorted(faces)

    def verify(self):
        objects={}
        for name,target in [('mug',np.array([.13,3.37,.84])),('block',np.array([.13,3.82,.87]))]:
            pos=self.d.body(name).xpos.copy()
            contacts=[]
            for c in self.d.contact:
                body_names=[self.m.body(int(self.m.geom_bodyid[g])).name for g in c.geom]
                geom_names=[self.m.geom(int(g)).name for g in c.geom]
                if name in body_names:contacts.extend(geom_names)
            supported='destination_top' in contacts
            released=not any('_grip_' in n for n in contacts)
            upright=float(self.d.body(name).xmat.reshape(3,3)[2,2])>.95
            speed=float(np.linalg.norm(self.d.qvel[self.m.joint(name+'_free').dofadr[0]:self.m.joint(name+'_free').dofadr[0]+6]))
            objects[name]={'position':pos.tolist(),'target':target.tolist(),'error_m':float(np.linalg.norm(pos-target)),
                           'supported_by_destination':supported,'released':released,'upright':upright,'speed':speed,
                           'passed':bool(np.linalg.norm(pos-target)<.06 and supported and released and upright and speed<.02)}
        return {'success':bool(self.done and all(o['passed'] for o in objects.values()) and not self.forbidden_contacts and not self.d.warning.number.any()),
                'simulation_seconds':float(self.d.time),'objects':objects,'events':self.events,
                'max_arm_ik_error_m':self.max_tracking_error,'minimum_carry_height_m':self.min_carry_height,
                'forbidden_contacts':self.forbidden_contacts,'warnings':self.d.warning.number.tolist()}

    def observe(self):
        """Validate every physics step, including contacts between controller updates."""
        m,d=self.m,self.d
        if not np.isfinite(d.qpos).all() or d.warning.number.any() or d.qpos[2]<.6:
            raise RuntimeError(f'Unstable simulation during {self.phase.name}')
        for c in d.contact:
            bodies=[int(m.geom_bodyid[g]) for g in c.geom]
            if any(b in self.robot_bodies for b in bodies):
                other=[m.body(b).name for b in bodies if b not in self.robot_bodies]
                if any(n in ('source','destination','barrier') for n in other) and c.dist<-.001:
                    pair=[m.body(b).name for b in bodies]
                    if pair not in self.forbidden_contacts:self.forbidden_contacts.append(pair)

    def update(self, dt=.01):
        if self.done:return
        m,d=self.m,self.d
        p=self.phase
        elapsed=d.time-self.started
        walking=p.goal is not None
        goal=p.goal if walking else self.walker.reference[:2].copy()
        reached=self.walker.update(d,goal,dt,walking)
        if p.pose is not None or p.carry:
            a=np.clip(elapsed/max(p.duration*.85,.001),0,1) if not walking else 1.
            a=a*a*(3-2*a)
            for s,y in [('left',.22),('right',-.23)]:
                if p.carry:
                    target=d.qpos[:3]+np.array([.72,y,.52])
                else:
                    target=np.array([p.pose[0],y+(3.6 if p.destination else 0),p.pose[1]])
                target=(1-a)*self.starts[s]+a*target
                q,ep,er=self.ik.solve(d,s,target,SIDE_GRASP,self.seeds.get(s),iterations=15)
                self.max_tracking_error=max(self.max_tracking_error,ep)
                self.seeds[s]=q
                qa,dof,act,site=self.ik.ids[s]
                # Gravity compensation through position targets, still capped by actuator torque limits.
                q=q+d.qfrc_bias[dof]/m.actuator_gainprm[act,0]
                d.ctrl[act]=np.clip(q,m.actuator_ctrlrange[act,0],m.actuator_ctrlrange[act,1])
        for s in ('left','right'):
            d.ctrl[m.actuator(s+'_grip_fingers_actuator').id]=255 if p.closed else 0
        if 7<=self.index<=12:
            for name in ('mug','block'):
                z=float(d.body(name).xpos[2])
                self.min_carry_height[name]=min(self.min_carry_height[name],z)
                if z<1.:
                    raise RuntimeError(f'{name} dropped during {p.name} (height {z:.3f})')
        advance=(reached and elapsed>1) if walking else elapsed>=p.duration
        if walking and elapsed>80:raise RuntimeError(f'Navigation timeout during {p.name}')
        if advance:
            if p.name=='close both grippers':
                for side,name in [('left','block'),('right','mug')]:
                    faces=self.pad_contacts(side,name)
                    if len(faces)!=2:raise RuntimeError(f'{name} has no bilateral pad grasp: {faces}')
                self.events.append({'phase':'bilateral grasps confirmed','time':round(d.time,3)})
            if self.index==len(PHASES)-1:
                self.done=True
                self.success=self.verify()['success']
                return
            self.index+=1
            self.started=d.time
            self.starts=self.pinch_positions()
            self.announce()
