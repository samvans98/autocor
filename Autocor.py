
import os
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, isnan
from multiprocessing import Pool
from functools import partial
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2


def readfile(inputarray):
    ### Set to greyscale if needed
    try:
        inputarray = cv2.cvtColor(inputarray, cv2.COLOR_RGB2GRAY)
    except Exception:
        pass

    return inputarray


class AutocorrelationTool:
    def __init__(self, inputimg):

        self.analysismode = None                    # choice from full or restricted
        self.restrictzone = None                    # should be [-90,90] when analysis mode = full

        self.inputimg = inputimg                    # select napari layer
        self.pxpermicron = None                     # import from napari metadata???
        self.gridsplitmode = None                   # choice from none, auto and manual
        self.gridsplitvalue = None                  # When gridsplitmode = auto input size of desired grid eg [100,100]
                                                    # When gridsplitmode = manual input number of grids eg [2,3]
        self.thresholdvalue = None
        self.autocorrelationmode = None             # choice from numpy or Miso



    def readfile(self):
        ### Set to greyscale if needed
        try:
            self.inputarray = cv2.cvtColor(self.inputimg, cv2.COLOR_RGB2GRAY)
        except Exception:
            pass

        self.inputarray = np.array(self.inputarray, dtype='float32')


    def medianthreshold(self):
        if self.inputarray == None:
            inputarray = self.readfile()

        temparray = np.array(self.inputarray, dtype='uint8')

        self.thresholdvalue = np.median(temparray)

    def preprocess(self):
        self.readfile()












### USER INPUT ###

def preprocess(inputarray):

    ### Set to greyscale if needed
    try:
        inputarray = cv2.cvtColor(inputarray, cv2.COLOR_RGB2GRAY)
    except Exception:
        pass

    ### Set type, blur and threshold

    inputarray = np.array(inputarray, dtype='float32')

    blurredz = np.array(inputarray, dtype='uint8')
    blurredz = cv2.GaussianBlur(blurredz, (5, 5), 5)

    blurredzoriginal = copy.deepcopy(blurredz)

    callproceed = False
    thresh = np.median(blurredz)

    ### MANUAL THRESHOLDING ###

    while not callproceed:
        blurredz = copy.copy(blurredzoriginal)
        thresholdH = blurredz[:, :] > thresh
        thresholdL = blurredz[:, :] <= thresh
        blurredz[thresholdH] = 1
        blurredz[thresholdL] = 0

        edged = cv2.Canny(blurredz, 0, 1)

        contours, hierarchy = cv2.findContours(blurredz, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        mask = np.zeros(blurredz.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

        cv2.imshow("img", mask)
        cv2.waitKey()

        print("the current treshold is ", thresh)
        ready = input("Proceed with this threshold? y/n     ")

        if ready == "n":
            thresh = float(input("Enter new threshold      "))
        else:
            callproceed = True

    ### overlay original image with mask
    inputarray[mask == 0] = np.nan

    return inputarray


def autocorr(x, method):
    def misoformula(list, index):

        def DEVSQ(list):
            minmean = np.array(list) - np.average(list)
            return np.sum(minmean ** 2)

        count = len(list) / 3
        arraylen = len(list) - count
        static = np.array(list[0:int(arraylen - 1)]) - np.average(list)
        dynamic = np.array(list[index:int(arraylen + index - 1)]) - np.average(list)
        sumproduct = np.sum(static * dynamic)
        return sumproduct / DEVSQ(list)

    if method == "numpy":
        result = np.correlate(x, x, mode='same')
        return result
    elif method == "miso":
        resultplus = []
        for index, val in enumerate(x):
            count = len(x) / 3
            if index < count:
                resultplus.append(misoformula(x, index))

        resultmin = np.flip(resultplus[1:])
        result = np.concatenate((resultmin, resultplus), axis=None)
        return result


def PrincipleComponents(df, mode, highlight):
    features = ["deg", "periodicity", "repeat"]
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    if mode == "2d":
        pca = PCA(n_components=2)

        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf.reset_index(), df.reset_index()], axis=1)
        print(finalDf)

        finalDf["periodicity"] = finalDf["periodicity"].astype(float)
        finalDf["repeat"] = finalDf["repeat"].astype(float)

        traceDf = finalDf[finalDf["repeat"].between(highlight[0], highlight[1])]

        fig = px.scatter(finalDf, x="principal component 1", y='principal component 2',
                         hover_data=["gridindex", "periodicity", "deg", "repeat"], color="periodicity")
        fig.add_traces(
            go.Scatter(x=traceDf["principal component 1"], y=traceDf['principal component 2'], mode="markers",
                       marker_symbol='star', marker_size=15, hoverinfo="none")
        )

    elif mode == "3d":

        df["periodicity"] = df["periodicity"].astype(float)
        df["repeat"] = df["repeat"].astype(float)

        traceDf = df[df["repeat"].between(highlight[0], highlight[1])]

        fig = px.scatter_3d(df, x="deg", y='repeat', z='periodicity',
                            hover_data=["gridindex", "periodicity", "deg", "repeat"], color="periodicity")
        fig.add_traces(
            go.Scatter3d(x=traceDf["deg"], y=traceDf['repeat'],
                         z=traceDf['periodicity'], mode="markers",
                         marker_symbol='diamond', marker_size=15, hoverinfo="skip")
        )

    fig.show()


def corr_pearson(x, y):
    """
    Compute Pearson correlation.
    """

    x_mean = np.nanmean(x)

    y_mean = np.nanmean(y)

    x1 = (x - x_mean) ** 2
    y1 = (y - y_mean) ** 2

    xsum = np.nansum(x1)
    ysum = np.nansum(y1)

    SP = np.nansum((x - x_mean) * (y - y_mean))

    r = SP / (sqrt(xsum) * sqrt(ysum))

    return r


def sinefunction(x, freq, phase):
    return 0.5 + 0.5 * np.sin(x * (2 * np.pi / freq) + phase)


def midline(matrix, angle):
    midpoint = ((np.array(np.shape(matrix)) - 1) / 2).astype(int)
    r = np.tan(np.radians(angle))
    e = midpoint[0] - (r * midpoint[1])
    if r == 0:
        intersectx = np.inf
    else:
        intersectx = -e / r
    intersecty = e
    if 0 <= intersecty <= np.shape(matrix)[0] - 1:
        intersect = [int(intersecty), 0]
        intersectoposite = [(np.shape(matrix)[0] - 1) - int(intersecty), np.shape(matrix)[1] - 1]
    else:
        intersect = [0, int(intersectx)]
        intersectoposite = [np.shape(matrix)[0] - 1, (np.shape(matrix)[1] - 1) - int(intersectx)]

    return intersect, intersectoposite


def extractline(matrix, start, end):
    # -- Extract the line...
    # Make a line with "num" points...
    y0, x0 = start  # These are in _pixel_ coordinates!!
    y1, x1 = end
    length = int(np.hypot(x1 - x0, y1 - y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    # Extract the values along the line
    zi = matrix[y.astype('int'), x.astype('int')]
    return zi


def checker(matrixshape, start, end):
    if start[0] == 0 and start[0] == end[0]:
        return False
    elif start[0] == matrixshape[0] - 1 and start[0] == end[0]:
        return False
    elif start[1] == 0 and start[1] == end[1]:
        return False
    elif start[1] == matrixshape[1] - 1 and start[1] == end[1]:
        return False
    else:
        return True


def movevertical(point, vector):
    if vector[0] > 0:
        point[0] = point[0] + 1
    elif vector[0] < 0:
        point[0] = point[0] - 1

    return point


def movehorizontal(point, vector):
    if vector[1] > 0:
        point[1] = point[1] + 1
    elif vector[1] < 0:
        point[1] = point[1] - 1

    return point


def nextpointsmart(matrixshape, deg, start, end, direction):
    vectormin90 = [np.sin(np.radians(deg - 90)), np.cos(np.radians(deg - 90))]
    vectorplus90 = [np.sin(np.radians(deg + 90)), np.cos(np.radians(deg + 90))]

    limr = matrixshape[0] - 1
    limc = matrixshape[1] - 1

    if direction == 'plus':
        vector = vectorplus90
    else:
        vector = vectormin90

    appendo = []

    for point in [start, end]:
        realoriginalpoint = copy.deepcopy((point))
        originalpoint = copy.deepcopy(point)
        if point[0] in [0, limr]:
            proposedpoint = movehorizontal(point, vector)
            if proposedpoint[1] > limc or proposedpoint[1] < 0:
                newproposedpoint = movevertical(originalpoint, vector)
                if newproposedpoint[0] > limr or newproposedpoint[0] < 0:
                    appendo.append(realoriginalpoint)
                else:
                    appendo.append(newproposedpoint)
            else:
                appendo.append(proposedpoint)
        elif point[1] in [0, limc]:
            proposedpoint = movevertical(point, vector)
            if proposedpoint[0] > limr or proposedpoint[0] < 0:
                newproposedpoint = movehorizontal(originalpoint, vector)
                if newproposedpoint[1] > limc or newproposedpoint[1] < 0:
                    appendo.append(realoriginalpoint)
                else:
                    appendo.append(newproposedpoint)
            else:
                appendo.append(proposedpoint)

    return appendo[0], appendo[1]


def gridsplit(array, mode, val):
    def internalsplit(array, val):
        if val[0] == 0:
            val[0] = 1
        if val[1] == 0:
            val[1] = 1
        array = np.transpose(array)
        slices = np.array_split(array, val[1])
        grids = []
        for slice in slices:
            slice = np.transpose(slice)
            grido = np.array_split(slice, val[0])
            for grid in grido:
                grids.append(grid)
        return grids

    val = val.split(",")
    print(val)
    val = list(map(int, val))

    if mode == 'custom':
        return internalsplit(array, val)

    if mode == 'auto':
        arrayshape = np.shape(array)
        rowsplit = int(arrayshape[0] / val[0])
        colsplit = int(arrayshape[1] / val[1])
        return internalsplit(array, [rowsplit, colsplit])


def nanarraycleaner(list):
    list = np.array(list, dtype='float32')

    dellist = []
    i = 0
    j = len(list) - 1
    while isnan(list[i]):
        dellist.append(i)
        i = i + 1

    while isnan(list[j]):
        dellist.append(j)
        j = j - 1

    index_set = set(dellist)  # optional but faster
    output = [x for i, x in enumerate(list) if i not in index_set]

    return output


def cycledegrees(input, pxpermicron, filename):
    grid = input[1]
    index = input[0]
    fitlist = []
    tempdict = {}
    dfPC = pd.DataFrame(columns=["deg", "periodicity", "repeat", "gridindex"])
    for deg in tqdm(range(-90, 90)):

        tempdict[deg] = {}

        intersect, intersectoposite = midline(grid, deg)

        originalintersect = copy.deepcopy(intersect)
        originalintersectoposite = copy.deepcopy(intersectoposite)

        x0 = intersect[1]
        x1 = intersectoposite[1]
        y0 = intersect[0]
        y1 = intersectoposite[0]

        meanarray = []

        while checker(np.shape(grid), intersect, intersectoposite):
            meanarray.append(np.nanmean(extractline(grid, intersect, intersectoposite)))
            intersect, intersectoposite = nextpointsmart(np.shape(grid), deg, intersect, intersectoposite, 'plus')

        while checker(np.shape(grid), originalintersect, originalintersectoposite):
            meanarray.insert(0, np.nanmean(extractline(grid, originalintersect, originalintersectoposite)))
            originalintersect, originalintersectoposite = nextpointsmart(np.shape(grid), deg, originalintersect,
                                                                         originalintersectoposite, 'min')

        newmeanarray = (np.array(meanarray) - np.nanmin(meanarray)) / (np.nanmax(meanarray) - np.nanmin(meanarray))

        newmeanarray = nanarraycleaner(newmeanarray)

        autocorlist = autocorr(newmeanarray, "miso")

        cormin = (np.diff(np.sign(np.diff(autocorlist))) > 0).nonzero()[0] + 1  # local min
        cormax = (np.diff(np.sign(np.diff(autocorlist))) < 0).nonzero()[0] + 1  # local max
        if len(cormax) < 3 or len(cormin) < 2:
            periodicity = np.nan
            repeat = np.nan
            fitlist.append([deg, periodicity, repeat])
        else:
            maxpoint = autocorlist[cormax[np.where(cormax == np.argmax(autocorlist))[0] + 1]]
            minpoint = autocorlist[cormin[int(len(cormin) / 2)]]

            periodicity = maxpoint - minpoint
            repeat = (cormax[np.where(cormax == np.argmax(autocorlist))[0] + 1] - len(autocorlist) / 2) / pxpermicron
            fitlist.append([deg, periodicity[0], repeat[0]])

        tempdict[deg] = {
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "intensityplot": newmeanarray,
            "autocorrelationplot": autocorlist,
            "cormin": cormin,
            "cormax": cormax,

        }

        if np.isnan(periodicity) or np.isnan(repeat):
            pass
        else:
            tempdf = pd.DataFrame({"deg": [deg], "periodicity": [periodicity[0]], "repeat": [repeat[0]],
                                   "gridindex": [filename + " / " + str(index)]})
            dfPC = pd.concat([dfPC, tempdf])

    fitlist = np.array(fitlist, dtype="float32")

    try:
        maxdeg = fitlist[np.nanargmax(fitlist[:, 1]), 0]
        repeatatmaxdeg = fitlist[np.nanargmax(fitlist[:, 1]), 2]

    except ValueError:
        maxdeg = 0
        repeatatmaxdeg = 0

    fig, axes = plt.subplots(nrows=3)
    axes[0].imshow(grid)
    axes[0].plot([tempdict[maxdeg]["x0"], tempdict[maxdeg]["x1"]], [tempdict[maxdeg]["y0"], tempdict[maxdeg]["y1"]],
                 'ro-')
    axes[0].axis('image')

    cormin = tempdict[maxdeg]["cormin"]
    cormax = tempdict[maxdeg]["cormax"]
    autocorlist = tempdict[maxdeg]["autocorrelationplot"]
    micronlist = np.array(range(len(autocorlist))) / pxpermicron

    axes[1].plot(tempdict[maxdeg]["intensityplot"])
    axes[2].plot(micronlist, autocorlist)
    axes[2].plot(cormin / pxpermicron, autocorlist[cormin], "o", label="min", color='r')
    axes[2].plot(cormax / pxpermicron, autocorlist[cormax], "o", label="max", color='b')
    axes[2].text(0.05, 0.95, np.nanmax(fitlist[:, 1]), transform=axes[2].transAxes, fontsize=14,
                 verticalalignment='top')
    axes[2].text(0.05, 0.7, repeatatmaxdeg, transform=axes[2].transAxes, fontsize=14,
                 verticalalignment='top')

    try:
        os.mkdir("output/" + filename)
    except FileExistsError:
        pass

    plt.savefig("output/" + filename + "/" + str(index) + ".jpg")

    return np.nanmax(fitlist[:, 1]), np.count_nonzero(np.isnan(grid)), repeatatmaxdeg, dfPC


# -- Generate some data...
# x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
# z = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)


if __name__ == '__main__':

    ### USER INPUT ###

    # print("Make sure you are running this script in the source directory and all .tif files are located in the input folder")
    #
    # pxpermicron = input("Enter pixel per micron value of image:     ")
    #
    # gridsplitmode = input("Splitmode: type either 'auto' or 'manual':       ")

    # if gridsplitmode not in ['auto', 'manual']:
    #     exit('Splitmode should be auto or manual')
    # else:
    #     if gridsplitmode == 'auto':
    #         gridsplitval = input("Enter the desired grid size as 'height, width' in pixels:     ")
    #     else:
    #         gridsplitval = input(
    #             "Enter the desired split as 'number of vertical slices, number of horizontal slices':     ")

    pxpermicron = 50

    gridsplitmode = "auto"

    gridsplitval = "100,100"

    ### Iterate over all file in input folder ###

    for filename in os.listdir("input"):
        f = os.path.join("input", filename)

        z = plt.imread(f)

        z = preprocess(z)

        cleangrids = []

        grids = gridsplit(z, gridsplitmode, gridsplitval)
        for index, grid in enumerate(grids):
            if not np.isnan(grid).all():
                cleangrids.append(grid)

        indexgrids = []

        for index, grid in enumerate(cleangrids):
            indexgrids.append([index, grid])

        with Pool(4) as pool:
            output = pool.map(partial(cycledegrees, pxpermicron=int(pxpermicron), filename=filename), indexgrids)
            output = np.array(output)
            weighted_avg = np.average(output[:, 0], weights=output[:, 1])
            intervallist = output[:, 2]
            medianrepeat = np.average(output[:, 2], weights=output[:, 1])
            print('FINAL RESULT', weighted_avg)
            print(intervallist)
            print('most likely periodicity interval', medianrepeat)

            df = pd.concat(output[:, 3])
            PrincipleComponents(df, "2d", (0.17, 0.21))


