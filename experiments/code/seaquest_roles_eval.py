import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np
import minatar_seaquest as task
from seaquest_report import paired
from wsplab.runners.torch import TorchRunner
H = Path.home() / '.wsplab/seaquest'
runner = TorchRunner(task.build, 'cpu'); seeds = list(task.EVAL_SEEDS)
def ret(w): return task.play(runner.call(w), seeds)[0][:, 0]
s = json.load(open(H / 'roles1/peers/run/search.json'))
out = {"original": ret(H / 'fixture/model.safetensors')}
for c in s['candidates']:
    out['roles/' + c['editor'].split(':')[0]] = ret(Path(c['folder']) / 'result/model.safetensors')
t1 = {c['editor'].split(':')[0]: c for c in json.load(open(H / 'trial1/independent/run/search.json'))['candidates']}
p1 = {c['editor'].split(':')[0]: c for c in json.load(open(H / 'trial1/peers/run/search.json'))['candidates']}
out['claude_solo'] = ret(Path(t1['anthropic']['folder']) / 'result/model.safetensors')
out['plain_council_pick'] = ret(Path(p1['openai']['folder']) / 'result/model.safetensors')
pick = 'roles/' + s['candidates'][s['selection']['run']]['editor'].split(':')[0]
res = {"means": {k: float(v.mean()) for k, v in out.items()}, "role_pick": pick,
       "role_pick_minus_claude_solo": paired(out[pick], out['claude_solo']),
       "role_pick_minus_plain_council": paired(out[pick], out['plain_council_pick']),
       "role_pick_minus_original": paired(out[pick], out['original']),
       "best_role_candidate_minus_claude_solo": paired(max((v for k, v in out.items() if k.startswith('roles/')), key=np.mean), out['claude_solo'])}
json.dump(res, open(H / 'roles1/comparison.json', 'w'), indent=2); print(json.dumps(res, indent=1))
