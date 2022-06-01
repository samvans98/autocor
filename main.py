import numpy as np
import cv2
import czifile
import os
import scipy.stats
from math import sqrt
import matplotlib.pyplot as plt

def findcor(file):
    img = czifile.imread("C:/Users/Sam Vanspauwen/Downloads/Emil/" + file)
    print(file)
    print(img.shape)

    Nav = img[0, 0, 1, 0, 0, :, :, 0]
    Chr = img[0, 0, 2, 0, 0, :, :, 0]

    Navarray = np.array(Nav, dtype="float32")
    Chrarray = np.array(Chr, dtype="float32")

    Chrblur = cv2.GaussianBlur(Chrarray, (5, 5), 5)

    ret, Chrthresh = cv2.threshold(Chrblur, 0.05 * 65534, 65534, cv2.THRESH_BINARY)

    cv2.imshow('ok', Chrthresh)
    cv2.waitKey()

    whitecount = np.count_nonzero(Chrthresh == 1)

    Navarray[Chrthresh == 0] = [np.nan]
    Chrarray[Chrthresh == 0] = [np.nan]

    cor = corr_pearson(Navarray, Chrarray, whitecount)

    return cor

def corr_pearson(x, y, count):

    """
    Compute Pearson correlation.
    """

    x_mean = np.nanmean(x)

    y_mean = np.nanmean(y)

    x1 = (x - x_mean)**2
    y1 = (y - y_mean)**2

    xsum = np.nansum(x1)
    ysum = np.nansum(y1)

    SP = np.nansum((x - x_mean)*(y - y_mean))

    r = SP / (sqrt(xsum)*sqrt(ysum))


    return r

filelist = []
ChR2 = []
C614 = []
C614KV = []

for file in os.listdir("C:/Users/Sam Vanspauwen/Downloads/Emil"):
    filelist.append(file)

for file in filelist:
    if "614-Kv" in file:
        C614KV.append(file)

    elif "614" in file:
        C614.append(file)

    else:
        # ChR2.append(file)



ChR2list = []
for file in ChR2:
    ChR2list.append(findcor(file))

C614list = []
for file in C614:
    C614list.append(findcor(file))

C614KVlist = []
for file in C614KV:
    C614KVlist.append(findcor(file))

data = [ChR2list, C614list, C614KVlist]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(data, patch_artist=True,
                notch='True', vert=0)

colors = ['#0000FF', '#00FF00',
          '#FFFF00']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#8B008B',
                linewidth=1.5,
                linestyle=":")

# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color='#8B008B',
            linewidth=2)

# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color='red',
               linewidth=3)

# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker='D',
              color='#e7298a',
              alpha=0.5)

# x-axis labels
ax.set_yticklabels(['data_1', 'data_2',
                    'data_3'])

# Adding title
plt.title("Customized box plot")

# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# show plot
plt.show()








