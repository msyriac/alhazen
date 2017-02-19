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

halo = True
delensTolerance = 1.0

snrange = np.arange(80,2100,200)
fsky = 0.4

gradCut = 10000
beamY = 1.5
noiseX = 3.0#*2.
noiseY = 3.0
noiseTX = noiseX
noisePX = np.sqrt(2.)*noiseTX
noiseTY = noiseY
noisePY = np.sqrt(2.)*noiseTY

tellminX = 100
tellmaxX = 3000
pellminX = 100
pellmaxX = 5000

lkneeTX,alphaTX = (350, -4.7)
lkneePX,alphaPX = (60, -2.6)
lkneeTY,alphaTY = (3400, -4.7)
lkneePY,alphaPY = (330, -3.8)
# lkneeTX,alphaTX = (0, -4.7)
# lkneePX,alphaPX = (0, -2.6)
# lkneeTY,alphaTY = (0, -4.7)
# lkneePY,alphaPY = (0, -3.8)


tellminY = 300
tellmaxY = 3000
pellminY = 200
pellmaxY = 5000


kmin = 40
deg = 10.
px = 0.5
#deg = 8.
#px = 0.8
dell = 10
cambRoot = "data/ell28k_highacc"
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)


frange = np.arange(2.,3100.,1.)
Clkk = theory.gCl("kk",frange)


#beamRange = np.arange(1.5,9.5,0.5)
#beamRange = np.arange(9.5,30.5,2.5)
#beamRange = np.arange(1.5,5.0,0.5)

beamX = 10.0
noiseRange = np.arange(3.0,30.0,4.0)

swap = False

#for polComb in ['TT','EB']:
for polComb in ['EB']:

    for delens in [False,True]:
        if polComb=='TT' and delens: continue
        pl = Plotter(scaleY='log',labelX="$L$",labelY="$C_L$")
        sns = []
        #for beamX in beamRange:
        for noiseTX in noiseRange:
            noisePX = np.sqrt(2.)*noiseTX

            myNls = NlGenerator(lmap,theory,gradCut=gradCut)

            kmax = getMax(polComb,tellmaxY,pellmaxY)


            bin_edges = np.arange(kmin,kmax,dell)+dell
            myNls.updateBins(bin_edges)


            if swap:
                tempB = beamY
                beamY = beamX
                beamX = tempB
                

            nTX,nPX,nTY,nPY = myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX, \
                              pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY, \
                              noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY, \
                              pellminY=pellminY,pellmaxY=pellmaxY,lkneesX=(lkneeTX,lkneePX), \
                              alphasX=(alphaTX,alphaPX), \
                              lkneesY=(lkneeTY,lkneePY),alphasY=(alphaTY,alphaPY))

            if polComb=='EB' and delens:
                ls, Nls,efficiency = myNls.iterativeDelens(polComb,delensTolerance,halo)
                print "percentage efficiency ", efficiency , " %"
            else:
                ls,Nls = myNls.getNl(polComb=polComb,halo=halo)
            
            pl.add(ls,Nls,label=str(beamX))
            #pl.add(myNls.N.cents,myNls.N.wxy,label=str(beamX))

            LF = LensForecast()
            LF.loadKK(frange,Clkk,ls,Nls)
            sn,errs = LF.sn(snrange,fsky,"kk")
            sns.append(sn)



        pl.add(frange,Clkk,color="black")

        #pl.legendOn(loc='lower left',labsize = 8)
        pl._ax.set_xlim(0,3000)
        pl._ax.set_ylim(1.e-9,1.e-6)
        pl.done("beamVary_"+polComb+"_delens_"+str(delens)+"_noiseVary.pdf")

        # pl = Plotter(labelX = "beamX (arcmin)",labelY="S/N auto",ftsize=14)
        # pl.add(beamRange,sns)
        # pl.done(polComb+str(delens)+"_sn_swap_"+str(swap)+"_noiseVary.pdf")

        pl = Plotter(labelX = "noiseX (muK-arcmin)",labelY="S/N auto",ftsize=14)
        pl.add(noiseRange,sns)
        pl.done(polComb+str(delens)+"_sn_swap_"+str(swap)+"_noiseVary.pdf")

