import matplotlib
matplotlib.use('Agg')
from orphics.tools.io import Plotter,dictFromSection,listFromConfig,getFileNameString
import flipper.liteMap as lm
from szlib.szcounts import ClusterCosmology
from alhazen.halos import NFWMatchedFilterSN
import numpy as np
from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax
import sys
from orphics.theory.gaussianCov import LensForecast
from orphics.tools.stats import bin2D
import os
import glob

halo = True
delensTolerance = 1.0

snrange = np.arange(80,2100,200)
fsky = 0.5

gradCut = 10000
#beamY = 1.5
beamY = 1.8
noiseY = 5./np.sqrt(2.)#3.0
noiseTY = noiseY
noisePY = np.sqrt(2.)*noiseTY

tellminX = 100
tellmaxX = 3000
pellminX = 100
pellmaxX = 5000

#lkneeTX,alphaTX = (350, -4.7)
#lkneePX,alphaPX = (60, -2.6)
#lkneeTY,alphaTY = (3400, -4.7)
#lkneePY,alphaPY = (330, -3.8)
lkneeTY,alphaTY = (0, 1)
lkneePY,alphaPY = (0, 1)



tellminY = 300
tellmaxY = 3000
pellminY = 200
pellmaxY = 5000


kmin = 40
deg = 10.
px = 0.5
dell = 10
cambRoot = "data/ell28k_highacc"
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)


frange = np.arange(2.,3100.,1.)
frangeC = np.arange(100.,5000.,20.)
Clkk = theory.gCl("kk",frange)

pl = Plotter(scaleY='log',labelX="$L$",labelY="$C_L$")

for polComb in ['EB']:

    #for delens,lines,alpha in zip([False,True],['-','--'],[1.0,0.5]):
    for delens,lines,alpha in zip([False],['-'],[1.0]):
        sns = []
        for noiseFile in ['fiducial']+glob.glob("data/colin*"):
        #for noiseFile in ['fiducial']+glob.glob("data/colin???.txt"):

            myNls = NlGenerator(lmap,theory,gradCut=gradCut)

            kmax = getMax(polComb,tellmaxY,pellmaxY)


            bin_edges = np.arange(kmin,kmax,dell)+dell
            myNls.updateBins(bin_edges)

            if "5m" in noiseFile:
                beamY = 1.8
                noiseY = 5./np.sqrt(2.)#3.0
                noiseTY = noiseY
                noisePY = np.sqrt(2.)*noiseTY
                
            elif "6m" in noiseFile:
                continue
                beamY = 1.5
                noiseY = 5./np.sqrt(2.)#3.0
                noiseTY = noiseY
                noisePY = np.sqrt(2.)*noiseTY
            elif noiseFile=="fiducial":
                pass
            else:
                raise ValueError
            
            if noiseFile == 'fiducial':
                beamX = beamY
                noiseTX = noiseTY #5./np.sqrt(2.)#3.
                noisePX = noisePY #np.sqrt(2.)*noiseTX
                lkneeTX,alphaTX = (0, 1)
                lkneePX,alphaPX = (0, 1)
                labname = noiseFile
                noiseFile = None
                lw = 2
            else:
                lw = 1
                beamX = 20
                noiseTX = 200
                noisePX = None
                lkneeTX = None
                lkneePX = None
                alphaTX = None
                alphaPX = None
                labname = os.path.basename(noiseFile)[:-4]
                
                
                

                

            nTX,nPX,nTY,nPY = myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX, \
                              pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY, \
                              noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY, \
                              pellminY=pellminY,pellmaxY=pellmaxY,lkneesX=(lkneeTX,lkneePX), \
                              alphasX=(alphaTX,alphaPX), \
                             lkneesY=(lkneeTY,lkneePY),alphasY=(alphaTY,alphaPY), noiseFilePX = noiseFile,noiseFilePY=noiseFile)
                                                
            if polComb=='EB' and delens:
                ls, Nls,efficiency = myNls.iterativeDelens(polComb,delensTolerance,halo,verbose=False)
                #print "percentage efficiency ", efficiency , " %"
            else:
                ls,Nls = myNls.getNl(polComb=polComb,halo=halo)
            
            # if True: #noiseFile is not None:
            #     #ls,Nls = np.loadtxt(noiseFile,unpack=True)
            #     #pl.add(ls,Nls,label=labname,ls=lines,alpha=alpha,lw=lw)
            #     binner = bin2D(myNls.N.modLMap,frangeC)
            #     lcents,Nlbinned = binner.bin(nPX)
            #     pl.add(lcents,Nlbinned,label=labname,ls="--")

            # pl.add(ls,Nls,label=labname,ls=lines,alpha=alpha)
                

            

            LF = LensForecast()
            LF.loadKK(frange,Clkk,ls,Nls)
            sn,errs = LF.sn(snrange,fsky,"kk")
            sns.append(sn)
            print noiseFile, " S/N " , sn



            
# pl.add(frangeC,theory.lCl('EE',frangeC))
# pl.legendOn(loc='lower right',labsize = 8)
# #pl._ax.set_xlim(0,3000)
# #pl._ax.set_ylim(1.e-9,1.e-6)
# pl.done("beamVary_"+polComb+".pdf")

            
# pl.add(frange,Clkk,color="black")
# pl.legendOn(loc='lower right',labsize = 8)
# pl._ax.set_xlim(0,3000)
# pl._ax.set_ylim(1.e-9,1.e-6)
# pl.done("beamVary_"+polComb+".pdf")
