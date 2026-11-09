import hard2_net as h
clean=h.accuracy(h.trained()); base=h.accuracy(h.broken())
print('clean',round(clean,4),'broken',round(base,4))
names={k:'+'.join(f.__name__ for f in v) for k,v in h.FAULTS.items()}
for d in h.FAULTS:
    g=h.accuracy(h.broken(skip={d}))-base
    print(f'repair digit {d} alone ({names[d]}): {100*g:+.1f} points')
order=sorted(h.FAULTS,key=lambda d:h.accuracy(h.broken(skip={d})),reverse=True)
skip=set(); path=[]
for d in order:
    skip.add(d); path.append(round(h.accuracy(h.broken(skip=skip)),3))
print('cumulative, best first',list(zip(order,path)))
d=7
net=h.trained()
import torch
with torch.no_grad():
    for k,f in h.FAULTS.items():
        for g in ([f[0]] if k==7 else f): g(net.branches[k])
print('digit 7 half repaired (rewire left, flip fixed)',round(100*(h.accuracy(net)-base),1))
