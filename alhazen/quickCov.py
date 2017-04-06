import matplotlib
matplotlib.use('Agg')
import numpy as np
import itertools
import flipper.liteMap as lm
import flipper.fftTools as ft
import orphics.analysis.flatMaps as fmaps
import orphics.tools.io as io
from orphics.tools.stats import bin2D
import sys
import os
import time
import orphics.tools.cmb as cmb
from scipy.interpolate import interp1d
from numpy.fft import fftshift

from alhazen.quadFunctions_old import fXY,F,crossIntegrand,WXY,WY

#polCombList=['TT','EB','EE','TB']
#polCombList=['TT','EB','TE','ET','EE','TB']
#polCombList=['TT','EB','TE','EE','TB']
polCombList=['TE','ET']
#polCombList=['TT']


halo = True
num_ells = 50 #300

# beamArcmin = 1.0
# noiseT = 1.0
# #noiseT = 5.0
# noiseP = np.sqrt(2.)*noiseT
# cmbellmin = 50
# cmbellmax = 8000
# kellmin = 2
# kellmax = 8000
# gradCut = None

#hu reproduce
beamArcmin = 7.0
noiseT = 27.0
noiseP = 56.6
cmbellmin = 50
cmbellmax = 3000
kellmin = 2
kellmax = 2000
gradCut = None


degx = 5.
degy = 5.
px = 2.0





TCMB = 2.7255e6

hugeTemplate = lm.makeEmptyCEATemplate(degx,degy,pixScaleXarcmin=px,pixScaleYarcmin=px)
lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(hugeTemplate)

nfreq = modLMap.max()
assert nfreq>cmbellmax
assert nfreq>kellmax

from orphics.tools.cmb import loadTheorySpectraFromCAMB
cambRoot = os.environ['HOME']+"/repos/cmb-lensing-projections/data/TheorySpectra/ell28k_highacc"
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=True,useTotal=False,TCMB = 2.7255e6,lpad=9000)
#theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000)

Nlfuncdict={}
Nlfuncdict['TT'] = cmb.get_noise_func(beamArcmin,noiseT,TCMB=TCMB)
Nlfuncdict['EE'] = cmb.get_noise_func(beamArcmin,noiseP,TCMB=TCMB)
Nlfuncdict['BB'] = cmb.get_noise_func(beamArcmin,noiseP,TCMB=TCMB)


print modLMap.shape
dlx = np.diff(lxMap,axis=1)[0,0]
dly = np.diff(lyMap,axis=0)[0,0]


Lmin = kellmin #10.
Lmax = kellmax #2000.
#Ls = np.logspace(np.log10(Lmin),np.log10(Lmax),num_ells)
Ls = np.linspace(Lmin,Lmax,num_ells)

lx1 = lyMap
ly1 = lxMap
lx1sq = lx1**2.
ly1sq = ly1**2.
l1sq = lx1sq+ly1sq
l1 = np.sqrt(l1sq)
ly2 = -ly1.copy()
ly2sq = ly2**2.
phi_l1 = np.arctan2(lx1,ly1)    





Als = {}
for polComb in polCombList:
    Als[polComb] = []

st = time.time()
for polComb in polCombList:
    XY = polComb
    X,Y = XY
    YY = Y+Y


    for L in Ls:

        lx2 = L - lx1
        lx2sq = lx2**2.
        l2sq = lx2sq+ly2sq
        l2 = np.sqrt(l2sq)
        Ll1 = L*lx1
        Ll2 = L*lx2

        phi_l2 = np.arctan2(lx2,ly2)    


        if polComb!='TT':
            cosDelta = np.cos(2.*(phi_l1-phi_l2))
            sinDelta = np.sin(2.*(phi_l1-phi_l2))
        else:
            cosDelta = None
            sinDelta = None

        f = fXY(XY,theory,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta)
            

        if True:
            if (XY in ['TE','ET']):
                fS = fXY(XY,theory,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=-sinDelta)
            else:
                fS = None
            
            Falpha = F(XY,f,fS,theory,Nlfuncdict,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta,halo=halo,gradCut=gradCut)
            Falpha[l1<cmbellmin]=0.
            Falpha[l1>cmbellmax]=0.
            Falpha[l2<cmbellmin]=0.
            Falpha[l2>cmbellmax]=0.
        
        integral = (Falpha*f).sum()*dlx*dly
        Alinv = integral/((2.*np.pi)**2.)/L**2.
        Als[polComb].append(1./(Alinv))



for polComb in polCombList:
    Als[polComb] = np.array(Als[polComb])
print time.time()-st," seconds."

crosses = {}
polCrosses = itertools.combinations_with_replacement(polCombList,2)



for alpha,beta in polCrosses:
    print alpha,beta
    Xalpha,Yalpha = alpha
    Xbeta,Ybeta = beta

    combs1 = [Xalpha+Xbeta,Yalpha+Ybeta]
    combs2 = [Xalpha+Ybeta,Yalpha+Xbeta]
    Cllist = ['TT','TE','EE','BB','ET']
    if not( all([combs in Cllist for combs in combs1])) and not(all([combs in Cllist for combs in combs2]) ):
        print "skipping"
        continue
    
    

    crosses[alpha+beta] = []
    for L in Ls:

        lx2 = L - lx1
        lx2sq = lx2**2.
        l2sq = lx2sq+ly2sq
        l2 = np.sqrt(l2sq)
        Ll1 = L*lx1
        Ll2 = L*lx2

        
        phi_l2 = np.arctan2(lx2,ly2)    


        cosDelta = np.cos(2.*(phi_l1-phi_l2))
        sinDelta = np.sin(2.*(phi_l1-phi_l2))
        

        falpha = fXY(alpha,theory,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta)
        fbeta = fXY(beta,theory,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta)
            

        falphaS = fXY(alpha,theory,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=-sinDelta)
        fbetaS = fXY(beta,theory,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=-sinDelta)

        Falpha = F(alpha,falpha,falphaS,theory,Nlfuncdict,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta,halo=halo,gradCut=gradCut)
        Falpha[l1<cmbellmin]=0.
        Falpha[l1>cmbellmax]=0.
        Falpha[l2<cmbellmin]=0.
        Falpha[l2>cmbellmax]=0.
        Fbeta = F(beta,fbeta,fbetaS,theory,Nlfuncdict,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta,halo=halo,gradCut=gradCut)
        Fbeta[l1<cmbellmin]=0.
        Fbeta[l1>cmbellmax]=0.
        Fbeta[l2<cmbellmin]=0.
        Fbeta[l2>cmbellmax]=0.
        FbetaS = F(beta,fbetaS,fbeta,theory,Nlfuncdict,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=-sinDelta,halo=halo,gradCut=gradCut)
        FbetaS[l1<cmbellmin]=0.
        FbetaS[l1>cmbellmax]=0.
        FbetaS[l2<cmbellmin]=0.
        FbetaS[l2>cmbellmax]=0.


        integral = crossIntegrand(alpha,beta,theory,Nlfuncdict,Falpha,Fbeta,FbetaS,l1,l2).sum()*dlx*dly
        N = integral/((2.*np.pi)**2.)
        crosses[alpha+beta].append(N)
    crosses[alpha+beta] = Als[alpha]*Als[beta]*np.array(crosses[alpha+beta])/4.
    
# pl = io.Plotter(scaleY='log',scaleX='log')
# pl.add(Ls,Als*Ls**2./4.)
# pl.done("nltt.png")



ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    

pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)
for polComb in polCombList:

    try:
        huFile = os.environ['HOME']+'/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_'+polComb.lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')
    except:
        huFile = os.environ['HOME']+'/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_'+polComb[::-1].lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')


    base_line, = pl.add(Ls,4.*crosses[polComb+polComb]/2./np.pi,ls="--",label=polComb)
    base_line, = pl.add(huell,hunl,ls='-.',label=polComb)
    #pl.add(Ls,4.*Als[polComb]*Ls**2./2./np.pi/4.,color=base_line.get_color())
    #pl.add(huell,hunl,ls='-.',color=col,label=polComb)

pl.legendOn(loc='lower left',labsize=10)
pl.done("testbin.png")




pl = io.Plotter()#scaleX='log')
polCrosses = itertools.combinations_with_replacement(polCombList,2)
for alpha,beta in polCrosses:
    if alpha==beta: continue
    Xalpha,Yalpha = alpha
    Xbeta,Ybeta = beta

    combs1 = [Xalpha+Xbeta,Yalpha+Ybeta]
    combs2 = [Xalpha+Ybeta,Yalpha+Xbeta]
    Cllist = ['TT','TE','EE','BB','ET']
    if not( all([combs in Cllist for combs in combs1])) and not(all([combs in Cllist for combs in combs2]) ):
        print "skipping"
        continue

    corr = crosses[alpha+beta]/np.sqrt(crosses[alpha+alpha]*crosses[beta+beta])
    base_line, = pl.add(Ls,corr,label=alpha+'x'+beta)
pl._ax.axhline(y=1.0,ls="--",color='black',alpha=0.4)
pl.legendOn(loc='lower left',labsize=10)
pl.done("crosses.png")



