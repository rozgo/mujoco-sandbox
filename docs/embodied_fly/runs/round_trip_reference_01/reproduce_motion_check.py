"""Read-only motion audit; preserve the original failed raw-speed gate."""
import json
from pathlib import Path
import numpy as np
from embodied_fly.provenance import sha256
source=Path('outputs/embodied_fly/round_trip_reference_02')
r=json.loads((source/'report.json').read_text())
assert sha256(source/'capture.npz')==r['capture_sha256']
s=np.load(source/'capture.npz')
rows=[]
for i,c in enumerate(r['cases']):
    ids=np.flatnonzero((s['time']>=11)&(s['time']<11.9))
    drift=(s['actual_position'][ids+50,i]-s['actual_position'][ids,i])*10/.1
    raw=s['actual_velocity'][s['time']>=11,i]*10
    tracking=all(w['position_peak_mm']<.5 for w in c['windows'].values())
    rows.append({'case':c['case'],'original_combined_gate':c['passed'],
        'all_intermediate_and_return_position_gates':tracking,
        'final_hold_raw_velocity_rms_xyz_mm_s':np.sqrt(np.mean(raw**2,axis=0)).tolist(),
        'final_hold_100ms_displacement_speed_max_mm_s':float(np.linalg.norm(drift,axis=1).max()),
        'final_hold_100ms_displacement_speed_rms_mm_s':float(np.sqrt(np.mean(np.sum(drift**2,axis=1)))),
        'no_physical_failure':c['first_failure_seconds'] is None})
report={'source_report_sha256':sha256(source/'report.json'),'capture_sha256':r['capture_sha256'],
    'original_speed_gate_failed':True,'raw_speed_limit_mm_s_original':1.5,
    'scope':'Supplementary analysis of the same capture, not a replacement or relabeling of its failed raw-speed gate. Displacement over 100ms measures sustained drift while raw 500Hz velocities retain rapid wingbeat vibration. No filtering or controller changes in live physics.',
    'stationary_reference_also_fails_original_speed_gate':not r['cases'][-1]['passed'],
    'cases':rows}
Path('docs/embodied_fly/runs/round_trip_reference_01/motion_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(rows,indent=2))
