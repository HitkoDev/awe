import enum
import json

import matplotlib
import matplotlib.pyplot as plt

with open('report.json', 'r+') as json_file:
    data = json.load(json_file)

fp = data['fp']
tp = data['tp']
total = data['total']

fp.sort()
tp.sort()

x = []
y = []
for i in range(len(fp)):
    pos = [p for p in tp if p >= fp[i]]
    num_fp = len(fp) - i
    num_tp = len(pos)
    num_fn = total - num_tp
    tp_rate = num_tp / total
    x.append(num_fp)
    y.append(tp_rate)


fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='False positive', ylabel='True positive rate')
ax.grid()

fig.savefig("report/figures/plot.png")
plt.show()
