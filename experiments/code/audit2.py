import json,os,re,sys
import numpy as np, torch
import hard_net
from wsplab.io import Checkpoint
from wsplab.runners.torch import mlp
R=os.path.expanduser('~/.wsplab/live/hard')
rows=[json.loads(l) for l in open(R+'/council_1/session/tool_calls.jsonl')]
for i in (3,6,11):
    print(f'== script {i}: lines touching inputs or X:')
    for ln in rows[i]['code'].splitlines():
        if re.search(r'inputs|X_dev|X\b|\bx\b|@',ln): print('   ',ln.strip()[:120])
net,clean,broken,F=hard_net.build()
x,y,tr,pr,ev=hard_net.data()
torch.manual_seed(11); c=mlp(hard_net.SIZES,"relu"); opt=torch.optim.Adam(c.parameters(),5e-3)
for _ in range(500):
    opt.zero_grad(); torch.nn.functional.cross_entropy(c(torch.tensor(x[tr])),torch.tensor(y[tr])).backward(); opt.step()
cs={k:v.detach().numpy() for k,v in c.state_dict().items()}
good=[j for j in range(48) if j not in F['wired']]
for run in ('council_1','claude_1','geminib_0','terra_1','noself_1'):
    e=Checkpoint(R+'/'+run+'/result/model.safetensors'); w0,w2,b2=e.float('layers.0.weight'),e.float('layers.2.weight'),e.float('layers.2.bias')
    fl=sum(np.corrcoef(w2[u,good],cs['layers.2.weight'][u,good])[0,1]>.9 for u in F['flipped'])
    wi=sum(np.abs(w2[:,u]-cs['layers.2.weight'][:,u]).min()<1e9 and abs(np.corrcoef(np.abs(w2[:,u]),np.abs(cs['layers.2.weight'][:,u]))[0,1])>.98 for u in F['wired'])
    mi=sum(np.abs(w0[u]-cs['layers.0.weight'][u]).max()<1e-2 for u in F['mirrored'])
    sh=sum(abs(b2[u]-cs['layers.2.bias'][u])<.5 for u in F['shifted'])
    print(run,'shifted',sh,'/12 flipped',fl,'/14 wired',wi,'/16 mirrored',mi,'/20')
