from orphics.tools.output import Plotter
import flipper.liteMap as lm
from szlib.szcounts import ClusterCosmology,dictFromSection,listFromConfig
from alhazen.halos import NFWMatchedFilterSN
import numpy as np
from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax
import sys


def getFileName(listOfNames,listOfVals):
    fullstr = ""
    for name,val in zip(listOfNames,listOfVals):
        fullstr += "_"+name+"_"+str(val)

    return fullstr



saveRoot = "data/ell28"

# Make a CMB Noise Curve
cambRoot = "data/ell28k_highacc"
halo = True
#delensTolerance = None
delensTolerance = 1.0

noiseY = float(sys.argv[1])
tellminY = int(sys.argv[2])
pellminY = int(sys.argv[3])

beamscale = lambda b: np.sqrt(8.*np.log(2.))*60.*180./np.pi/b

kmin = 40
deg = 10.
#px = 0.2
px = 0.5
dell = 10
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)
print lmap.data.shape

i=0
for gradCut in [10000,2000]:
    myNls = NlGenerator(lmap,theory,gradCut=gradCut)
    for polComb in ['TT','TE','EE','EB','ET','TB']:
        for beamY in np.arange(0.5,10.0,0.5):
            beamell = beamscale(beamY)
            for tellmaxY,pellmaxY in [(3000,5000),(beamell,beamell)]:

                for noiseX,beamX,lab,tellminX,tellmaxX,pellminX,pellmaxX in \
                    [(noiseY,beamY,"sameGrad",tellminY,tellmaxY,pellminY,pellmaxY), \
                     (30.,7.0,"planckGrad",2,3000,2,3000)]:

                    noiseTX = noiseX
                    noisePX = np.sqrt(2.)*noiseTX
                    noiseTY = noiseY
                    noisePY = np.sqrt(2.)*noiseTY



                    kmax = getMax(polComb,tellmaxY,pellmaxY)
                    i+=1
                    print i,tellmaxY,pellmaxY,kmax

                    bin_edges = np.arange(kmin,kmax,dell)+dell
                    myNls.updateBins(bin_edges)
                    myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX, \
                                      pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY, \
                                      noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY, \
                                      pellminY=pellminY,pellmaxY=pellmaxY)
                    
                    if (polComb!='EB' and polComb!='TB') or (delensTolerance is None):
                        ls,Nls = myNls.getNl(polComb=polComb,halo=halo)
                    else:
                        ls, Nls = myNls.iterativeDelens(polComb,delensTolerance,halo)



                    fileName = saveRoot + getFileName(['gradCut','polComb','beamY','noiseY','grad','tellminY','pellminY','kmin','deg','px','delens'],[gradCut,polComb,beamY,noiseY,lab,tellminY,pellminY,kmin,deg,px,delensTolerance])+".txt"
                    np.savetxt(fileName,np.vstack((ls,Nls)).transpose())
