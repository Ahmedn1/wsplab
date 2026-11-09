import numpy as np, torch, os, json
from sklearn.datasets import load_digits
from wsplab.runners.torch import mlp, state_to_checkpoint
out=os.path.expanduser("~/.wsplab/live/agent_task")
x,y=load_digits(return_X_y=True); x=(x/16).astype(np.float32); y=y.astype(np.int64)
o=np.random.default_rng(7).permutation(len(x)); x,y=x[o],y[o]
torch.manual_seed(7); net=mlp([64,32,10],"relu"); opt=torch.optim.Adam(net.parameters(),1e-2)
for _ in range(400):
    opt.zero_grad(); torch.nn.functional.cross_entropy(net(torch.tensor(x[:900])),torch.tensor(y[:900])).backward(); opt.step()
clean=(net(torch.tensor(x[1300:])).argmax(1).numpy()==y[1300:]).mean()
with torch.no_grad():
    w=net.layers[2].weight
    top=w[6].abs().argsort(descending=True)[:6]   # the units class 6 relies on most now feed the wrong sign into it
    w[6,top]*=-1
broken=(net(torch.tensor(x[1300:])).argmax(1).numpy()==y[1300:]).mean()
state_to_checkpoint(net.state_dict(),out+"/net.safetensors")
np.savez(out+"/practice.npz",inputs=x[900:1300],targets=y[900:1300]); np.savez(out+"/heldout.npz",inputs=x[1300:],targets=y[1300:])
print(float(clean),float(broken)); 
