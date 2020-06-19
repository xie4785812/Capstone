import pwlf
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show,legend,savefig
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import segment
import fit

client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze

msft = db['NVDA']
cursor = msft.find({})

data = []
for i in cursor:
    data.append(float(i['price'].replace(',','')))

data = np.array(data)
data = data[0:200]
time = np.array([i for i in range(len(data))])

max_error = 0.5

def draw_segments(segments,data):
    ax = gca()
    min = np.min(data)
    max = np.max(data)
    ax.set_ylim([min-1,max+1])
    code = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,Path.CLOSEPOLY]
    for segment in segments:
        x0 = segment[0]
        x1 = segment[2]
        y0 = segment[1]
        y1 = segment[3]
        line = Line2D([segment[0],segment[2]],[segment[1],segment[3]])
        ax.add_line(line)

        ax.scatter(segment[2], segment[3],marker = '^',color = 'black',s = 40)
        vertx = [(x0,y1),(x0,y0),(x1,y0),(x1,y1),(0,0)]
        path = Path(vertx,code)
        if y1 > y0:
            patch = patches.PathPatch(path, facecolor='none', lw = 0.5)
            ax.add_patch(patch)
        elif y1 < y0:
            patch = patches.PathPatch(path, facecolor='gray', lw=0.5)
            ax.add_patch(patch)
        else:
            dis = x1-x0
            v = [(x0,y1-2*dis),(x0,y0+2*dis),(x1,y0+2*dis),(x1,y1-2*dis),(0,0)]
            path = Path(v,code)
            patch = patches.PathPatch(path, facecolor='black', lw=0.5)
            ax.add_patch(patch)

def draw_plot(data,plot_title):
    plot(range(len(data)),data,alpha=0.8,color='red')
    title(plot_title)
    xlabel("Time seres in 5 second skip(s)")
    ylabel("Price($)")
    xlim((0,len(data)-1))
    legend()

figure()
segments = segment.topdownsegment(data, fit.interpolate, fit.sumsquared_error, max_error)
draw_plot(data, 'top-down with simple interpolation')
draw_segments(segments,data)
savefig('status box.png')
show()


