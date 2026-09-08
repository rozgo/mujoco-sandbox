"""End-to-end physics validation of the approved two-object task."""
import mujoco
import numpy as np

from sixlegs.control import LEGS,leg_ik
from sixlegs.scene import load_scene
from sixlegs.simulation import Simulation


def test_leg_ik_matches_forward_kinematics():
    m,d=load_scene()
    rng=np.random.default_rng(17)
    for name in LEGS:
        mount=m.body(name+'_coxa').pos.copy()
        sign=1 if name.startswith('left') else -1
        index=('front','middle','rear').index(name.split('_')[1])
        angle=sign*(np.pi/4+index*np.pi/4)
        for _ in range(10):
            q=rng.uniform([-.25,-.15,-.2],[.25,.15,.2])
            for j,value in zip(('yaw','hip','knee'),q):d.joint(name+'_'+j).qpos[0]=value
            mujoco.mj_forward(m,d)
            recovered=leg_ik(d.site(name+'_toe').xpos-d.qpos[:3],mount,angle)
            np.testing.assert_allclose(recovered,q,atol=1e-7)


def test_complete_physical_transfer():
    sim=Simulation()
    m,d=sim.model,sim.data
    assert m.nmocap==0
    assert m.neq==6  # Only the two grippers' linkage constraints, no object attachment.
    assert not np.any(m.eq_type==mujoco.mjtEq.mjEQ_WELD)
    max_y=-np.inf
    min_x=np.inf
    update=sim.demo.update
    def checked_update(dt):
        before=d.qpos.copy()
        update(dt)
        np.testing.assert_array_equal(d.qpos,before)
    sim.demo.update=checked_update
    while not sim.demo.done and d.time<150:
        sim.step()
        assert not np.any(d.xfrc_applied)
        assert not np.any(d.qfrc_applied)
        max_y=max(max_y,float(d.qpos[1]))
        min_x=min(min_x,float(d.qpos[0]))
    report=sim.report()
    assert sim.demo.done and report['success'],report
    assert min_x < -2.6 and max_y>3.55
    assert report['maximum_torque_fraction']<=1.00001
    assert report['forbidden_contacts']==[]
    assert report['minimum_carry_height_m']['mug']>1.
    assert report['minimum_carry_height_m']['block']>1.
    assert all(o['passed'] for o in report['objects'].values())
    assert any(e['phase']=='bilateral grasps confirmed' for e in report['events'])
