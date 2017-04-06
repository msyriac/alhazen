from alhazen.quadraticEstimator import isotropic_noise_full_lensing_covariance
from orphics.tools.cmb import loadTheorySpectraFromCAMB
import sys,os
import numpy as np
from orphics.tools.io import Plotter
import orphics.tools.cmb as cmb

TCMB = 2.7255e6
outDir = os.environ['WWW']
cambRoot = "/astro/u/msyriac/repos/cmb-lensing-projections/data/TheorySpectra/ell28k_highacc"
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=True,useTotal=False,TCMB = TCMB,lpad=9000)

ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    


polCombList = ['TT','TE','EE','TB','EB']
colorList = [None]*len(polCombList)

kellmin = 2
kellmax = 3000
num_ells = 50
px = 2.0

beam = 7.0
noiseT = 27.0
noiseP = 56.5
tellmin = 2.
tellmax = 3000.

noiseFuncTX = cmb.get_noise_func(beam,noiseT,ellmin=tellmin,ellmax=tellmax,TCMB=TCMB)
noiseFuncEX = cmb.get_noise_func(beam,noiseP,ellmin=tellmin,ellmax=tellmax,TCMB=TCMB)
noiseFuncTY = noiseFuncTX
noiseFuncEY = noiseFuncBY = noiseFuncBX = noiseFuncEX

Ls,crosses = isotropic_noise_full_lensing_covariance(polCombList,theory,noiseFuncTX,noiseFuncEX,noiseFuncBX,noiseFuncTY,noiseFuncEY,noiseFuncBY,kellmin,kellmax,num_ells,independentExperiments=False,degx = 5.,degy = 5.,px = px,TCMB = TCMB)

pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

# CHECK THAT NORM MATCHES HU/OK
for polComb,col in zip(polCombList,colorList):

    try:
        huFile = 'data/hu_'+polComb.lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')
    except:
        huFile = 'data/hu_'+polComb[::-1].lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')

    pl.add(Ls,4.*crosses[polComb+polComb]/2./np.pi,color=col)
    pl.add(huell,hunl,ls='--',color=col)


pl.done(outDir+"testbin.png")
