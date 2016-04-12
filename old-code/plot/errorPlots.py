from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from scipy import stats



fig = plt.figure()
ax = fig.add_subplot(111)
X_MIN = 0.0
X_MAX = 100.0
Y_MIN = 0.0
Y_MAX = 1.5
ax.set_title("MSE",fontsize=14,fontweight='bold')
ax.axis([X_MIN,X_MAX,Y_MIN,Y_MAX])
INVERSE_GOLDEN_RATIO=0.618
AXIS_ASPECT_RATIO=X_MAX/Y_MAX
ax.set_aspect(INVERSE_GOLDEN_RATIO*AXIS_ASPECT_RATIO)
ax.set_xlabel("iteration", fontsize=12)
ax.set_ylabel("MSE", fontsize=12)
series= ['Cocoa','Lbfgs','SGD']
colors=[(0.6,0.011,0.043), (0.258, 0.282, 0.725),(0.2117, 0.467, 0.216)]
pp = []
ss = []
for serie,i in zip(series,range(len(series))):
    file_in = open('/Users/mac/Desktop/summer_project/distributed-ML-benchmark/distributed-ML-benchmark/output/Iter_'+serie+'.txt','r')
    lines = file_in.readlines()
    nIter = []
    for line in lines:
        split_line = line.rstrip('\n').split(' ')
        nIter.append(split_line)
    file_in.close()
    xData = []
    for x in nIter:
        for x1 in x:
             xData.append(float(x1))

    nError = []
    file_in = open('/Users/mac/Desktop/summer_project/distributed-ML-benchmark/distributed-ML-benchmark/output/MSE_'+serie+'.txt','r')
    lines = file_in.readlines()
    for line in lines:
        split_line = line.rstrip('\n').split(' ')
        nError.append(split_line)
    file_in.close()
    yData = []
    for y in nError:
        for y1 in y:
            yData.append(float(y1))
    # first we'll do it the default way, with gaps on weekends
    p, = ax.plot(xData, yData, 'o-', ms= 1.5, mfc=colors[i], mec=colors[i], color=colors[i], label=serie)
    pp.append(p)
    ss.append(serie)
    # next we'll write a custom formatter

ax.legend(pp,ss, numpoints=1, loc='best',fontsize = 12).get_frame().set_visible(True)
ax.axhline(y=0.195, linewidth=0.75, color='black', linestyle='--')
label_string = "0.195"
ax.text(1 - len(label_string) / 60. - 0.01, 0.16, label_string, fontsize=12, transform=ax.transAxes)


fig.savefig("output.pdf", dpi=250,  bbox_inches='tight')