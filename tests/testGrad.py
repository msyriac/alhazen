import matplotlib
matplotlib.use('Agg')
from orphics.tools.io import Plotter, getFileNameString
import numpy as np

saveRoot = "data/ell28"

gradCut = 2000
polComb = 'TT'
beamY = 1.5
noiseY = 3.0
tellminY = 200
pellminY = 50
kmin = 40
deg = 10.
px = 0.2
delensTolerance = 1.0

pl = Plotter()

for lab in ['planckGrad','sameGrad']:
    fileName = saveRoot + getFileNameString(['gradCut','polComb','beamY','noiseY','grad','tellminY','pellminY','kmin','deg','px','delens'],[gradCut,polComb,beamY,noiseY,lab,tellminY,pellminY,kmin,deg,px,delensTolerance])+".txt"

    ells, Nls = np.loadtxt(fileName,unpack=True)

    pl.add(ells,Nls,label=lab)
pl.done("output/grad.png")
