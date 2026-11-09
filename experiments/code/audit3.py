import os, numpy as np, torch
import hard_net
from wsplab.io import Checkpoint
from wsplab.runners.torch import mlp
R=os.path.expanduser('~/.wsplab/live/hard')
net,clean,broken,F=hard_net.build()
x,y,tr,pr,ev=hard_net.data()
torch.manual_seed(11); c=mlp(hard_net.SIZES,"relu"); opt=torch.optim.Adam(c.parameters(),5e-3)
for _ in range(500):
    opt.zero_grad(); torch.nn.functional.cross_entropy(c(torch.tensor(x[tr])),torch.tensor(y[tr])).backward(); opt.step()
cw=c.state_dict()['layers.2.weight'].detach().numpy()
print('true flipped',sorted(F['flipped']))
for run in ('council_1','geminib_0','noself_1','claude_1'):
    e=Checkpoint(R+'/'+run+'/result/model.safetensors').float('layers.2.weight')
    match=np.isclose(e,cw,atol=2e-2).mean(1)          # share of entries in each row equal to the clean network
    fixed=[int(u) for u in F['flipped'] if match[u]>.6]
    healthy=[u for u in range(48) if u not in F['flipped']]
    print(run,'flipped rows matching clean',len(fixed),'/14; healthy rows matching clean',int((match[healthy]>.6).sum()),'/',len(healthy),'; mean match on flipped rows',round(float(match[F['flipped']].mean()),3))
print('neg list used by council_1 vs true: overlap',len(set([2,3,4,5,7,25,30,44,46,6,16,27,32,39])&set(int(u) for u in F['flipped'])))
