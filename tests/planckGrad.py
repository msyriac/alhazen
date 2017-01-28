import sys
import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time

from orphics.tools.output import Plotter
from ConfigParser import SafeConfigParser 

import flipper.liteMap as lm

from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax


from scipy.fftpack import fftshift,ifftshift,fftfreq
from pyfftw.interfaces.scipy_fftpack import fft2
from pyfftw.interfaces.scipy_fftpack import ifft2

from alhazen.halos import NFWkappa,getDLnMCMB,predictSN


beamX = 1.5
beamY = 1.5
noiseTX = 20.
noisePX = np.sqrt(2.)*noiseTX
noiseTY = 20.
noisePY = np.sqrt(2.)*noiseTY
tellminX = 1000
tellmaxX = 3000
pellminX = 1000
pellmaxX = 5000
tellminY = 1000
tellmaxY = 3000
pellminY = 1000
pellmaxY = 5000


polComb = 'TT'
kmin = 100

cambRoot = "data/ell28k_highacc"
gradCut = 10000
halo = True
saveId = ""


kmax = getMax(polComb,tellmaxY,pellmaxY)




# Make a CMB Noise Curve
deg = 10.
px = 0.5
dell = 10
bin_edges = np.arange(kmin,kmax,dell)+dell
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)
myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY,noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY,pellminY=pellminY,pellmaxY=pellmaxY)
ls,Nls = myNls.getNl(polComb=polComb,halo=halo)

ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    
pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)



pl.add(ls,4.*Nls/2./np.pi,label="act,act")


beamX = 7.0
noiseTX = 30.
noisePX = np.sqrt(2.)*noiseTX
tellminX = 2
tellmaxX = 3000
myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY,noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY,pellminY=pellminY,pellmaxY=pellmaxY)
ls,Nls = myNls.getNl(polComb=polComb,halo=halo)
pl.add(ls,4.*Nls/2./np.pi,label="planck,act")


noiseTX = 0.
myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY,noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY,pellminY=pellminY,pellmaxY=pellmaxY)
ls,Nls = myNls.getNl(polComb=polComb,halo=halo)
pl.add(ls,4.*Nls/2./np.pi,label="planckNoNoise,act")


pl.legendOn(loc='lower left',labsize=10)
pl.done("output/"+saveId+"nl.png")
