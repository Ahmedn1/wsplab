import json, os, re, numpy as np, torch
import hard_net
from wsplab.io import Checkpoint
from wsplab.runners.torch import mlp
R=os.path.expanduser('~/.wsplab/live/peers/pilot')
net,clean,broken,F=hard_net.build()
x,y,tr,pr,ev=hard_net.data()
torch.manual_seed(11); c=mlp(hard_net.SIZES,"relu"); opt=torch.optim.Adam(c.parameters(),5e-3)
for _ in range(500):
    opt.zero_grad(); torch.nn.functional.cross_entropy(c(torch.tensor(x[tr])),torch.tensor(y[tr])).backward(); opt.step()
cs={k:v.detach().numpy() for k,v in c.state_dict().items()}; bs={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
def acc(sd):
    m=mlp(hard_net.SIZES,"relu"); m.load_state_dict({k:torch.tensor(v) for k,v in sd.items()})
    return float((m(torch.tensor(x[ev])).argmax(1).numpy()==y[ev]).mean())
names=('layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','layers.4.weight','layers.4.bias')
cand={}
for k,who in enumerate(('claude','gemini','terra')):
    e=Checkpoint(f'{R}/run_{k}/result/model.safetensors'); sd={n:e.float(n) for n in names}; cand[who]=sd
    w0,w2,b2=sd['layers.0.weight'],sd['layers.2.weight'],sd['layers.2.bias']
    m=np.isclose(w2,cs['layers.2.weight'],atol=2e-2)
    print(who,'held-out',round(acc(sd),4),'shifted',sum(abs(b2[u]-cs['layers.2.bias'][u])<.5 for u in F['shifted']),'/12 flipped',int((m[F['flipped']].mean(1)>.6).sum()),'/14 wired',int((m[:,F['wired']].mean(0)>.6).sum()),'/16 mirrored',sum(np.abs(w0[u]-cs['layers.0.weight'][u]).max()<1e-2 for u in F['mirrored']),'/20')
    print('   changed vs broken: L0 rows',int((np.abs(w0-bs['layers.0.weight']).max(1)>1e-3).sum()),'L2 rows',int((np.abs(w2-bs['layers.2.weight']).max(1)>1e-3).sum()),'L2 cols',int((np.abs(w2-bs['layers.2.weight']).max(0)>1e-3).sum()),'L4 rows',int((np.abs(sd['layers.4.weight']-bs['layers.4.weight']).max(1)>1e-3).sum()))
print('clean',round(acc(cs),4))
g,cl=cand['gemini'],cand['claude']
for label,sd in (('claude layer 2 + gemini layer 0',{**cl,'layers.0.weight':g['layers.0.weight']}),('gemini without its output rows',{**g,'layers.4.weight':bs['layers.4.weight']}),('claude without its output rows',{**cl,'layers.4.weight':bs['layers.4.weight']})):
    print(label,round(acc(sd),4))
print('--- did peers use each other?')
for k,who in enumerate(('claude','gemini','terra')):
    T=sorted(os.listdir(f'{R}/run_{k}/session/turns')); n=0; ex=None
    for t in T:
        p=f'{R}/run_{k}/session/turns/{t}/answer.md'
        if os.path.exists(p):
            a=open(p).read()
            mm=re.search(r'[^.\n]*\b(peer|Peer [ABC])\b[^.\n]*\.',a)
            if mm: n+=1; ex=ex or re.sub(r'\s+',' ',mm.group(0))[:190]
    print(who,'replies citing a peer',n,'of',len(T),'|',ex)
