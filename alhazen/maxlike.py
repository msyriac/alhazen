import numpy as np


def lnlike(logdet,covinv,stamp):
    Npix = stamp.size
    assert covinv.size==Npix**2
    vec = stamp.reshape((Npix,1))
    quad = np.dot(np.dot(vec.T,covinv),vec)
    ans = logdet + quad
    assert ans.size==1
    return ans[0,0]
