import numpy as np

def DEVSQ(list):
    minmean = np.array(list) - np.average(list)
    return np.sum(minmean**2)


a = [50,47,52,46,45,48]

print(DEVSQ(a))

