
import numpy as np


N = 100
px = 0.5 #arcmin

trueArea = (N*px)**2.

print "true area ", trueArea , " arcmin^2"

mat = np.ones((N,N))
easyArea = np.sum(mat)*px**2.

print "easy area ", easyArea , " arcmin^2"

from numpy.fft import fft2,ifft2

ft = fft2(mat)
diffArea = ifft2(ft*ft).real * px**2. #/ N/N

print "difficult area ", diffArea , " arcmin^2"

