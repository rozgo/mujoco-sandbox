import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import detrend, find_peaks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frequency(x):
    spectrum = abs(np.fft.rfft(detrend(x) * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), .002)
    spectrum[f < 1] = 0
    return float(f[spectrum.argmax()])


def diagnose(source, name):
    p = source / f'{name}.npz'
    s = np.load(p)
    m = mujoco.MjModel.from_binary_path(str(source / 'model.mjb'))
    joints = m.actuator_trnid[14:20, 0]
    assert all('wing' in m.joint(int(j)).name for j in joints)
    qindices, vindices = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    window = (s['time'] >= 6) & (s['time'] < 10)
    q, v = s['qpos'][window], s['qvel'][window]
    h = (q[:, 2] - s['requested_height_cm'][window]) * 10
    peaks = find_peaks(h, distance=8, prominence=.01)[0]
    troughs = find_peaks(-h, distance=8, prominence=.01)[0]
    cycles = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        between = troughs[(troughs > a) & (troughs < b)]
        if len(between):
            cycles.append(float((h[a] + h[b]) / 2 - h[between].min()))
    full = s['time'] >= 1
    vertical_mm_s = s['qvel'][full, 2] * 10
    heights = (s['qpos'][full, 2] - s['requested_height_cm'][full]) * 10
    horizontal = np.linalg.norm(s['qpos'][full, :2] - s['requested_xy_cm'][full], axis=1) * 10
    horizontal_v = np.linalg.norm(s['qvel'][full, :2], axis=1) * 10
    physical_scores = {
        'height': float((2 / np.sqrt(1 + heights**2)).mean()),
        'horizontal_position': float((1 / np.sqrt(1 + horizontal**2)).mean()),
        'horizontal_velocity': float((1 / np.sqrt(1 + (horizontal_v / 5)**2)).mean()),
    }
    speed_scores = {str(scale): float((1 / np.sqrt(1 + (vertical_mm_s / scale)**2)).mean())
                    for scale in (50, 20)}
    return {
        'source_sha256': sha(p), 'window_seconds': [6, 10],
        'height_mean_error_mm': float(h.mean()), 'height_total_span_mm': float(np.ptp(h)),
        'height_dominant_hz': frequency(h),
        'height_ripple_mean_peak_to_trough_mm': float(np.mean(cycles)) if cycles else None,
        'height_cycles_measured': len(cycles),
        'sweep_dominant_hz': [frequency(q[:, qindices[j]]) for j in (0, 3)],
        'mean_absolute_sweep_velocity_rad_s': [float(abs(v[:, vindices[j]]).mean()) for j in (0, 3)],
        'sweep_angle_span_rad': [float(np.ptp(q[:, qindices[j]])) for j in (0, 3)],
        'mean_stroke_orientation_rad': [float(q[:, qindices[j]].mean()) for j in (1, 4)],
        'horizontal_world_velocity_mean_mm_s': (v[:, :2].mean(axis=0) * 10).tolist(),
        'offline_reward_terms_after_one_second': physical_scores,
        'vertical_speed_scores_by_scale_mm_s': speed_scores,
    }


old = Path('outputs/embodied_fly/hover_only_review_05')
new = Path('outputs/embodied_fly/hover_only_review_06')
a, b = read(old / 'report.json'), read(new / 'report.json')
assert a['physical_contract'] == b['physical_contract']
assert a['model_sha256'] == b['model_sha256']
assert next(r for r in a['results'] if r['case'] == 'pid') == next(r for r in b['results'] if r['case'] == 'pid')
report = {
    'method': 'Saved 500 Hz physical captures. t=[6,10)s for cycle statistics; detrend/Hann FFT, frequencies >=1Hz (.25Hz resolution). Peaks/troughs at least8 samples apart with .01mm prominence. Mean adjacent peak height minus intervening minimum. Reward terms use pre-action states t>=1s; four motion terms only, not complete return. All original physical gates stay unchanged.',
    'before_checkpoint_sha256': a['checkpoint_sha256'],
    'after_checkpoint_sha256': b['checkpoint_sha256'],
    'pid': diagnose(new, 'pid'),
    'before': diagnose(old, 'hover'),
    'after': diagnose(new, 'hover'),
    'evaluation_before': a['results'], 'evaluation_after': b['results'],
}
out = Path('docs/embodied_fly/runs/hover_only_06/comparison.json')
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('evaluation_before', 'evaluation_after')}, indent=2))
