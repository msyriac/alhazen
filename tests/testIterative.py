print "Starting imports..."
import matplotlib
matplotlib.use('Agg')
from alhazen.quadraticEstimator import Estimator, NlGenerator
import orphics.analysis.flatMaps as fmaps 
import numpy as np
import flipper.liteMap as lm
from orphics.tools.io import Plotter
from orphics.tools.stats import binInAnnuli
import sys,os

outDir = os.environ['WWW']

halo = True
beam = 1.0
noiseT = 1.0
noiseP = np.sqrt(2.)*noiseT
tellmin = 100
tellmax = 3000
gradCut = 10000

pellmin = 100
pellmax = 5000

deg = 5.
px = 1.0
arc = deg*60.

kellmin = 10
kellmax = 5000
bin_edges = np.arange(kellmin,kellmax,10)

from orphics.theory.cosmology import Cosmology
cc = Cosmology(lmax=int(max(tellmax,pellmax,kellmax)),pickling=True)
theory = cc.theory


lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)
myNls.updateNoise(beam,noiseT,noiseP,tellmin,tellmax,pellmin,pellmax)

polCombList = ['TT','EE','ET','EB','TB']
colorList = ['red','blue','green','orange','purple']
ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    

lsmv,Nlmv,ells,dclbb,efficiency = myNls.getNlIterative(polCombList,kellmin,kellmax,tellmax,pellmin,pellmax,dell=10,halo=True,plot=True)

print efficiency

pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

# CHECK THAT NORM MATCHES HU/OK
for polComb,col in zip(polCombList,colorList):
    ls,Nls = myNls.getNl(polComb=polComb,halo=halo)
    # try:
    #     huFile = 'data/hu_'+polComb.lower()+'.csv'
    #     huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')
    # except:
    #     huFile = 'data/hu_'+polComb[::-1].lower()+'.csv'
    #     huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')


    pl.add(ls,4.*Nls/2./np.pi,color=col)
    #pl.add(huell,hunl,ls='--',color=col)
pl.add(lsmv,4.*Nlmv/2./np.pi,color='black',lw=3,ls="--")

pl.done(outDir+"nltest.png")
