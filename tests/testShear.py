import matplotlib
matplotlib.use('Agg')
import os
from alhazen.shear import dndz
import numpy as np
from orphics.tools import io

ngalArcminSquare = 30.

zrange = np.arange(0.,3.0,0.1)

print np.trapz(dndz(zrange),zrange)

fracgals = []
for i,z in enumerate(zrange):
    fracgal = np.trapz(dndz(zrange[i:]),zrange[i:,])
    fracgals.append(fracgal)
    print z

pl = io.Plotter()
pl.add(zrange,fracgals)
outDir = os.environ['WWW']+"plots/"
pl.done(outDir + "fracgals.png")
