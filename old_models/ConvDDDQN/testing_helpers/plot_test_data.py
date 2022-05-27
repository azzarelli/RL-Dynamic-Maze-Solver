import matplotlib.pyplot as plt
import json
import numpy as np

DATA = []

changed = ['epdec0.1.json',
           'epdec0.01.json',
           'epdec0.001.json',
           'epdec0.0001.json'
           ]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


plt.figure(1)

cnt = 1
for c in changed:
    with open(c, 'r') as js:
        d = moving_average(json.loads(js.read()), n=100)
        plt.plot(d, label=str(float(cnt/10)))
        cnt += 1

plt.xlabel('Episodes')
plt.ylabel('Score')

plt.legend(title='Gamma')
plt.savefig('VaryingEpmin.png')
plt.show()


