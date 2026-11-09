import glob, json, os, re, numpy as np, torch
import hard_net
from wsplab.io import Checkpoint
from wsplab.runners.torch import mlp
R=os.path.expanduser('~/.wsplab/live/team')
net,clean,broken,F=hard_net.build()
x,y,tr,pr,ev=hard_net.data()
torch.manual_seed(11); c=mlp(hard_net.SIZES,"relu"); opt=torch.optim.Adam(c.parameters(),5e-3)
for _ in range(500):
    opt.zero_grad(); torch.nn.functional.cross_entropy(c(torch.tensor(x[tr])),torch.tensor(y[tr])).backward(); opt.step()
cs={k:v.detach().numpy() for k,v in c.state_dict().items()}
sim=re.compile(r"(inputs|X_dev|X\b|xs?\b)\s*@|@\s*(inputs|X_dev)|\.backward|lstsq|pinv|optim\.|enable_grad|requires_grad",re.I)
for d in sorted(glob.glob(R+'/*_[1-5]')):
    p=d+'/result/model.safetensors'
    if not os.path.exists(p): print(os.path.basename(d),'no result'); continue
    e=Checkpoint(p); w0,w2,b2=e.float('layers.0.weight'),e.float('layers.2.weight'),e.float('layers.2.bias')
    m=np.isclose(w2,cs['layers.2.weight'],atol=2e-2)
    fl=int((m[F['flipped']].mean(1)>.6).sum()); wi=int((m[:,F['wired']].mean(0)>.6).sum())
    sh=sum(abs(b2[u]-cs['layers.2.bias'][u])<.5 for u in F['shifted']); mi=sum(np.abs(w0[u]-cs['layers.0.weight'][u]).max()<1e-2 for u in F['mirrored'])
    code=[json.loads(l) for l in open(d+'/session/tool_calls.jsonl')]
    bad=[i for i,r in enumerate(code) if r.get('kind')=='python' and sim.search(r['code'])]
    rounds=[json.loads(l) for l in open(d+'/session/council.jsonl')] if os.path.exists(d+'/session/council.jsonl') else []
    print(os.path.basename(d),f'shifted {sh}/12 flipped {fl}/14 wired {wi}/16 mirrored {mi}/20 | suspect scripts {bad} | rounds',[(q['proposals'][q['winner']]['member'].split(':')[0],[round(t,1) for t in q['totals']]) for q in rounds])
