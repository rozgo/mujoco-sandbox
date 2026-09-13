import hashlib
import json
import shutil
import statistics
from pathlib import Path

import torch


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


source = Path('outputs/embodied_fly/hover_only_06')
target = Path('docs/embodied_fly/runs/hover_only_06')
review = Path('outputs/embodied_fly/hover_only_review_06')
report, evaluation = read(source / 'report.json'), read(review / 'report.json')
for a,b in (('report.json','training.json'),('progress.jsonl','progress.jsonl')):
    shutil.copy2(source/a,target/b)
shutil.copy2(review/'report.json', target/'evaluation.json')
asset = Path('assets/embodied_fly/diagnostics/hover_only_06.pt')
shutil.copy2(source/'actor.pt',asset)
parent_path = Path('assets/embodied_fly/diagnostics/hover_only_05.pt')
parent = torch.load(parent_path,map_location='cpu',weights_only=False)
child = torch.load(asset,map_location='cpu',weights_only=False)
assert sha(asset)==report['checkpoint_sha256']==evaluation['checkpoint_sha256']
assert sha(parent_path)==report['parent_checkpoint_sha256']==child['parent_checkpoint_sha256']
assert parent['physical_contract']==child['physical_contract']==evaluation['physical_contract']
assert parent['graph_sha256']==child['graph_sha256']
assert parent['graph_metadata_sha256']==child['graph_metadata_sha256']
assert sha(review/'model.mjb')==evaluation['model_sha256']==report['model_sha256']
old_recipe = read('docs/embodied_fly/runs/hover_only_05/training.json')['reward_recipe']
new_recipe = report['reward_recipe']
recipe_changes = {k:{'before':old_recipe.get(k),'after':new_recipe.get(k)}
                  for k in old_recipe.keys()|new_recipe.keys()
                  if old_recipe.get(k)!=new_recipe.get(k)}
assert recipe_changes=={'vertical_speed_scale_cm_s':{'before':5.0,'after':2.0}}, recipe_changes
fixed,changed = [],[]
for key,value in child['state_dict'].items():
    assert torch.isfinite(value).all()
    (fixed if torch.equal(value,parent['state_dict'][key]) else changed).append(key)
assert all(k in fixed for k in child['state_dict'] if k.startswith(('utility_head.','intention_encoder.','observation_')) or k.endswith('_ids'))
assert all('core.'+k in changed for k in ('excitability','leak','bias'))
optimizer_audit = {}
for key,count,fresh in (('optimizer_state_dict',report['ppo_updates'],False),
                        ('value_optimizer_state_dict',report['critic_updates'],True)):
    before,after = parent[key],child[key]
    for i,item in after['state'].items():
        old = 0 if fresh else int(before['state'][i]['step'])
        assert int(item['step'])-old==count,(key,i,old,item['step'],count)
    optimizer_audit[key]={'parameters_with_history':len(after['state']),
                         'step_increment':count,'fresh_for_changed_reward':fresh}
traces = 0
for e in report['completed_episodes']:
    if e.get('failure_trace'):
        trace=e['failure_trace']
        assert sha(source/trace['file'])==trace['sha256']
        traces+=1
write(target/'artifact_verification.json',{
    'checkpoint_sha256':sha(asset),'physical_graph_contracts_preserved':True,
    'all_actor_tensors_finite':True,'fixed_tensors':fixed,'changed_tensors':changed,
    'optimizer_history':optimizer_audit,'reward_recipe_changes':recipe_changes,
    'verified_failure_windows':traces,
})
rows = [json.loads(s) for s in (source/'progress.jsonl').read_text().splitlines()]
keys=read('docs/embodied_fly/runs/hover_only_05/MEASURED_STATS.json').keys()
stats={k:report[k] for k in keys if k in report}
stats.update(source_commit=report['provenance']['source_commit'],evaluation=evaluation['results'],
             median_max_kl=statistics.median(r['max_kl'] for r in rows if r['actor_updates_enabled']),
             maximum_kl=max(r['max_kl'] for r in rows if r['max_kl'] is not None),completed_episodes=len(report['completed_episodes']),
             failed_episodes=sum(e['failed'] for e in report['completed_episodes']),
             initial_std_range=[min(report['initial_exploration_std']),max(report['initial_exploration_std'])],
             critic_warmup=report['critic_warmup'],reward_recipe_changes=recipe_changes)
write(target/'MEASURED_STATS.json',stats)
print(json.dumps({k:v for k,v in stats.items() if k!='evaluation'},indent=2))
