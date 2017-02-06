from orphics.tools.io import Plotter,dictFromSection,listFromConfig,getFileNameString
import flipper.liteMap as lm
from szlib.szcounts import ClusterCosmology
from alhazen.halos import NFWMatchedFilterSN
import numpy as np
from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax
import sys





saveRoot = "data/px0d5/ell28"

# Make a CMB Noise Curve
cambRoot = "data/ell28k_highacc"
halo = True
#delensTolerance = None
delensTolerance = 1.0

noiseY = float(sys.argv[1])
tellminY = int(sys.argv[2])
pellminY = int(sys.argv[3])

#beamRange = np.arange(0.5,5.0,0.5)
beamRange = np.arange(5.0,10.5,0.5)

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
        for beamY in beamRange:
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
                    print i,tellmaxY,pellmaxY,kmax,"delens:",delensTolerance

                    bin_edges = np.arange(kmin,kmax,dell)+dell
                    myNls.updateBins(bin_edges)
                    myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX, \
                                      pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY, \
                                      noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY, \
                                      pellminY=pellminY,pellmaxY=pellmaxY)



                    
                    ls,Nls = myNls.getNl(polComb=polComb,halo=halo)

                    fileName = saveRoot + getFileNameString(['gradCut','polComb','beamY','noiseY','grad','tellminY','pellminY','tellmaxY','pellmaxY','kmin','deg','px'],[gradCut,polComb,beamY,noiseY,lab,tellminY,pellminY,tellmaxY,pellmaxY,kmin,deg,px])+".txt"
                    np.savetxt(fileName,np.vstack((ls,Nls)).transpose())

                    if (polComb=='EB' or polComb=='TB') and (delensTolerance is not None):
                        ls, Nls = myNls.iterativeDelens(polComb,delensTolerance,halo)
                        fileName = saveRoot + getFileNameString(['gradCut','polComb','beamY','noiseY','grad','tellminY','pellminY','tellmaxY','pellmaxY','kmin','deg','px','delens'],[gradCut,polComb,beamY,noiseY,lab,tellminY,pellminY,tellmaxY,pellmaxY,kmin,deg,px,delensTolerance])+".txt"
                        np.savetxt(fileName,np.vstack((ls,Nls)).transpose())
