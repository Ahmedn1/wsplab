import json,os,re,sys,glob
import numpy as np, torch
sys.path.insert(0,os.environ['LIVE_DIR'])
import hard_net
from wsplab.io import Checkpoint
R=os.path.expanduser('~/.wsplab/live/hard')
net,clean,broken,F=hard_net.build()
# the clean state: rebuild by training without sabotage
x,y,tr,pr,ev=hard_net.data()
torch.manual_seed(11)
from wsplab.runners.torch import mlp
c=mlp(hard_net.SIZES,"relu"); opt=torch.optim.Adam(c.parameters(),5e-3)
for _ in range(500):
    opt.zero_grad(); torch.nn.functional.cross_entropy(c(torch.tensor(x[tr])),torch.tensor(y[tr])).backward(); opt.step()
cs={k:v.detach().numpy() for k,v in c.state_dict().items()}; bs={k:v.detach().numpy() for k,v in net.state_dict().items()}
print('faults',{k:len(v) for k,v in F.items()})
def repaired(run):
    e=Checkpoint(R+'/'+run+'/result/model.safetensors')
    w0,w2,b2=e.float('layers.0.weight'),e.float('layers.2.weight'),e.float('layers.2.bias')
    close=lambda a,b:np.abs(a-b).max()<1e-2
    return {'mirrored':sum(close(w0[u],cs['layers.0.weight'][u]) for u in F['mirrored']),
            'wired':sum(close(w2[:,u],cs['layers.2.weight'][:,u]) for u in F['wired']),
            'flipped':sum(close(w2[u],cs['layers.2.weight'][u]) for u in F['flipped']),
            'shifted':sum(abs(b2[u]-cs['layers.2.bias'][u])<.5 for u in F['shifted'])}
for run in ('council_1','council_0','claude_1','geminib_0','terra_1','noself_1'):
    print(run,repaired(run))
d=R+'/council_1/session'
print(open(d+'/final_answer.md').read()[:1500])
rows=[json.loads(l) for l in open(d+'/tool_calls.jsonl')]
print('---- code audit for council_1')
sus=re.compile(r"inputs\.npy|np\.maximum|relu|@\s*W|softmax|argmax|torch\.nn|forward\(|lstsq|polyfit|corrcoef|gradient",re.I)
for i,r in enumerate(rows):
    if r.get('kind')=='python':
        c_=r['code']; hits=sorted(set(m.group(0).lower() for m in sus.finditer(c_)))
        print(i,len(c_),hits,'|',c_[:110].replace('\n',' / '))
    else:
        q=r.get('request','')[:90].replace('\n',' '); print(i,'FORWARD',q, (r.get('error') or '')[:60])
