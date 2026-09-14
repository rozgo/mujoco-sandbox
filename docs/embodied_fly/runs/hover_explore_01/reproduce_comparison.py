"""Reproduce this recorded trial comparison from its saved capture directories."""
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


before = Path('outputs/embodied_fly/pid_imitation_teacher_review_01')
after = Path('outputs/embodied_fly/hover_explore_review_01')
reports = [json.loads((p/'report.json').read_text()) for p in (before, after)]
assert reports[0]['model_sha256'] == reports[1]['model_sha256']
assert reports[0]['physical_contract'] == reports[1]['physical_contract']
assert reports[0]['results'][3] == reports[1]['results'][3]

cases, waveforms, relative = [], {}, {}
for i, name in enumerate(('hover', 'lower_start', 'lateral_start')):
    old, new = [r['results'][i] for r in reports]
    assert old['case'] == new['case'] == name
    cases.append({'case':name, 'before':old, 'after':new})
    waveforms[name] = {
        'before':diagnose(before, name) if old['stable'] else None,
        'after':diagnose(after, name) if new['stable'] else None,
    }
for label, path in [('before', before), ('after', after)]:
    baseline = np.load(path/'hover.npz')
    relative[label] = {}
    for name in ('lower_start', 'lateral_start'):
        s = np.load(path/(name+'.npz'))
        assert np.array_equal(s['time'], baseline['time'])
        delta = (s['qpos'][:,:3] - baseline['qpos'][:,:3])*10
        final = s['time'] >= 9
        relative[label][name] = {
            'initial_position_delta_mm':delta[0].tolist(),
            'initial_velocity_delta_mm_s':((s['qvel'][0,:3]-baseline['qvel'][0,:3])*10).tolist(),
            'last_second_mean_position_delta_mm':delta[final].mean(axis=0).tolist(),
            'last_second_relative_position_rms_mm':float(np.sqrt(np.mean(np.sum(delta[final]**2,axis=1)))),
            'interpretation':'Difference from the same actor nominal trajectory, which may itself drift. Small relative error alone is not target recovery.',
        }
old, new = [r['results'][0]['windows']['whole_capture'] for r in reports]
checks = {
    'all_three_starts_airborne':all(r['stable'] for r in reports[1]['results'][:3]),
    'nominal_position_rms_improves_10percent':new['position_rms_mm'] <= .9*old['position_rms_mm'],
    'nominal_altitude_rms_improves_10percent':new['altitude_rms_mm'] <= .9*old['altitude_rms_mm'],
}
for case in cases[1:]:
    a,b = [case[k]['windows']['whole_capture'] for k in ('before','after')]
    for metric in ('position_rms_mm','altitude_rms_mm'):
        checks[case['case']+'_'+metric+'_does_not_worsen_10percent'] = b[metric] <= 1.1*a[metric]
nominal = waveforms['hover']['after']
checks['nominal_sweep_frequency_25_to_35_hz'] = bool(nominal and all(25 <= f <= 35 for f in nominal['sweep_dominant_hz']))
checks['nominal_repeated_height_ripple_below_0_2mm'] = bool(nominal and nominal['height_ripple_mean_peak_to_trough_mm'] is not None and nominal['height_ripple_mean_peak_to_trough_mm'] < .2)
report = {
    'method':'Full-window RMS for declared progress criteria; original after-first-second and PID-quality gates retained. Waveforms from saved500Hz states at6-10s, detrended Hann FFT with.25Hz bins; mean adjacent peak height minus intervening trough for repeated ripple, as earlier pilot06. Failed trajectories receive no waveform-success inference.',
    'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'before_checkpoint_sha256':reports[0]['checkpoint_sha256'],
    'after_checkpoint_sha256':reports[1]['checkpoint_sha256'],
    'before_report_sha256':hashlib.sha256((before/'report.json').read_bytes()).hexdigest(),
    'after_report_sha256':hashlib.sha256((after/'report.json').read_bytes()).hexdigest(),
    'progress_checks':checks, 'development_progress_pass':all(checks.values()),
    'unchanged_accurate_hover_passes':sum(r['reference_quality_gate'] for r in reports[1]['results'][:3]),
    'cases':cases, 'waveforms':waveforms, 'perturbed_vs_nominal':relative,
    'pid':reports[1]['results'][3],
}
Path('docs/embodied_fly/runs/hover_explore_01/comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['progress_checks','development_progress_pass','unchanged_accurate_hover_passes']},indent=2))
for c in cases:print(c['case'], c['after']['windows']['after_first_second'])
print('nominal waveform',nominal)
