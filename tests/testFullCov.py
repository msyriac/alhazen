from alhazen.quadraticEstimator import isotropic_noise_full_lensing_covariance, NlGenerator
from orphics.tools.cmb import loadTheorySpectraFromCAMB
import sys,os
import numpy as np
from orphics.tools.io import Plotter
import orphics.tools.cmb as cmb
import flipper.liteMap as lm

TCMB = 2.7255e6
outDir = os.environ['WWW']
cambRoot = "/astro/u/msyriac/repos/cmb-lensing-projections/data/TheorySpectra/ell28k_highacc"
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=True,useTotal=False,TCMB = TCMB,lpad=9000)

ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    


polCombList = ['TT','TE','EE','TB','EB','ET']
#polCombList = ['EE','EB']
#polCombList = ['TE','ET']
colorList = [None]*len(polCombList)

kellmin = 100
kellmax = 5000
num_ells = 50
px = 1.5

halo = True
gradCut = 10000 #2000
# beam = 7.0
# noiseT = 27.0
# noiseP = 56.5
tellmin = 100.
tellmax = 3000.

pellmin = 100.
pellmax = 5000.

beam = 1.0
noiseT = 3.0
noiseP = np.sqrt(2.)*noiseT

noiseFuncTX = cmb.get_noise_func(beam,noiseT,ellmin=tellmin,ellmax=tellmax,TCMB=TCMB)
noiseFuncEX = cmb.get_noise_func(beam,noiseP,ellmin=pellmin,ellmax=pellmax,TCMB=TCMB)
noiseFuncTY = noiseFuncTX
noiseFuncEY = noiseFuncBY = noiseFuncBX = noiseFuncEX

Ls,Nls,crosses,Nmv = isotropic_noise_full_lensing_covariance(polCombList,theory,noiseFuncTX,noiseFuncEX,noiseFuncBX,noiseFuncTY,noiseFuncEY,noiseFuncBY,kellmin,kellmax,num_ells,spacing="linear",independentExperiments=False,degx = 5.,degy = 5.,px = px,TCMB = TCMB,halo=halo,gradCut=gradCut)

### Nlgenerator cross-check
deg = 10.
px = 0.5
arc = deg*60.
lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
bin_edges = np.arange(kellmin,kellmax,num_ells)
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)
myNls.updateNoise(beam,noiseT,noiseP,tellmin,tellmax,pellmin,pellmax)
####




pl = Plotter(scaleY='log')#,scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

# CHECK THAT NORM MATCHES HU/OK
for polComb,col in zip(polCombList,colorList):
    if polComb=='EB':
        lsold, Nlsold, eff = myNls.iterativeDelens(polComb,1.0,True)
    else:
        lsold,Nlsold = myNls.getNl(polComb=polComb,halo=halo)
    
    try:
        huFile = 'data/hu_'+polComb.lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')
    except:
        huFile = 'data/hu_'+polComb[::-1].lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')

    pl.add(Ls,4.*crosses[polComb+polComb]/2./np.pi,color=col,label=polComb)
    #pl.add(Ls,4.*Nls[polComb]/2./np.pi,color=col,alpha=0.2)
    pl.add(lsold,4.*Nlsold/2./np.pi,color=col,alpha=1.0,ls="-.")
    #pl.add(huell,hunl,ls='--',color=col)
pl.add(Ls,4.*Nmv/2./np.pi,color="black",alpha=1.0)

pl.legendOn(labsize=10)
pl.done(outDir+"testbin.png")


