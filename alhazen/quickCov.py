import matplotlib
matplotlib.use('Agg')
import numpy as np

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

#polCombList=['TT','EB','EE','TB']
polCombList=['TT','EB','TE','ET','EE','TB']


halo = False
num_ells = 500

# beamArcmin = 1.0
# noiseT = 1.0
# noiseP = 1.414
# cmbellmin = 50
# cmbellmax = 3000
# kellmin = 2
# kellmax = 4000
# gradCut = None

# hu reproduce
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


def fXY(XY,theory,ll1,ll2,l1,l2,cos2phi=None,sin2phi=None):
    C = theory.uCl

    if XY=='TT':
        return ll1*C('TT',l1)+ll2*C('TT',l2)
    elif XY=='TE':
        return ll1*cos2phi*C('TE',l1)+ll2*C('TE',l2)
    elif XY=='EE':
        return (ll1*C('EE',l1)+ll2*C('EE',l2))*cos2phi
    elif XY=='ET':
        return ll2*cos2phi*C('TE',l2)+ll1*C('TE',l1)
    elif XY=='EB':
        return ll1*C('EE',l1)*sin2phi
    elif XY=='TB':
        return ll1*C('TE',l1)*sin2phi

def F(XY,f,fS,theory,Nlfuncdict,ll1,ll2,l1,l2,cos2phi=None,sin2phi=None):
    X,Y = XY
    
    if XY in ['TT','EE']:
        #f = fXY(XY,theory,ll1,ll2,l1,l2,cos2phi=cos2phi,sin2phi=sin2phi)
        return 0.5*f*np.nan_to_num(1./(theory.lCl(X+X,l1)+Nlfuncdict[X+X](l1)))*np.nan_to_num(1./(theory.lCl(Y+Y,l2)+Nlfuncdict[Y+Y](l2)))
    elif XY in ['EB','TB']:
        #f = fXY(XY,theory,ll1,ll2,l1,l2,cos2phi=cos2phi,sin2phi=sin2phi)
        return f*np.nan_to_num(1./(theory.lCl(X+X,l1)+Nlfuncdict[X+X](l1)))*np.nan_to_num(1./(theory.lCl(Y+Y,l2)+Nlfuncdict[Y+Y](l2)))
    elif XY=='TE':
        #f = fXY(XY,theory,ll1,ll2,l1,l2,cos2phi=cos2phi,sin2phi=sin2phi)
        #fS = fXY(XY,theory,ll2,ll1,l2,l1,cos2phi=cos2phi,sin2phi=-sin2phi)

        C_EE = lambda ell: theory.lCl('EE',ell)+Nlfuncdict['EE'](ell)
        C_TT = lambda ell: theory.lCl('TT',ell)+Nlfuncdict['TT'](ell)
        C_TE = lambda ell: theory.lCl('TE',ell)
        
        C_EE_l1 = C_EE(l1)
        C_TT_l2 = C_TT(l2)
        C_TE_l1 = C_TE(l1)
        C_TE_l2 = C_TE(l2)

        prod1 = C_TE_l1*C_TE_l2
        prod2 = C_EE_l1*C_TT_l2

        return (prod2*f - prod1*fS) / ( (C_TT(l1)*C_EE(l2)*prod2) - (prod1*prod1))
    elif XY=='ET':
        #fS = fXY(XY,theory,ll1,ll2,l1,l2,cos2phi=cos2phi,sin2phi=sin2phi)
        #f = fXY(XY,theory,ll2,ll1,l2,l1,cos2phi=cos2phi,sin2phi=-sin2phi)
        ftemp = fS.copy()
        fS = f.copy()
        f = ftemp.copy()

        C_EE = lambda ell: theory.lCl('EE',ell)+Nlfuncdict['EE'](ell)
        C_TT = lambda ell: theory.lCl('TT',ell)+Nlfuncdict['TT'](ell)
        C_TE = lambda ell: theory.lCl('TE',ell)
        
        C_EE_l2 = C_EE(l2)
        C_TT_l1 = C_TT(l1)
        C_TE_l1 = C_TE(l1)
        C_TE_l2 = C_TE(l2)

        prod1 = C_TE_l2*C_TE_l1
        prod2 = C_EE_l2*C_TT_l1

        return (prod2*f - prod1*fS) / ( (C_TT(l2)*C_EE(l1)*prod2) - (prod1*prod1))

    
def WXY(XY,theory,Nlfuncdict,l1):

    X,Y = XY
    if Y=='B': Y='E'
    gradClXY = X+Y
    if XY=='ET': gradClXY = 'TE'
    W = np.nan_to_num(theory.uCl(gradClXY,l1)/(theory.lCl(X+X,l1)+Nlfuncdict[X+X](l1)))

    return W

def WY(YY,theory,Nlfuncdict,l2):
    assert YY[0]==YY[1]
    W = np.nan_to_num(1./(theory.lCl(YY,l2)+Nlfuncdict[YY](l2)))
    return W

    



TCMB = 2.7255e6

# degx = 15.
# degy = 15.
# px = 0.5
hugeTemplate = lm.makeEmptyCEATemplate(degx,degy,pixScaleXarcmin=px,pixScaleYarcmin=px)
lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(hugeTemplate)

from orphics.tools.cmb import loadTheorySpectraFromCAMB
cambRoot = "/home/msyriac/repos/cmb-lensing-projections/data/TheorySpectra/ell28k_highacc"
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=True,useTotal=False,TCMB = 2.7255e6,lpad=9000)
#theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000)

Nlfuncdict={}
Nlfuncdict['TT'] = cmb.get_noise_func(beamArcmin,noiseT,TCMB=TCMB)
Nlfuncdict['EE'] = cmb.get_noise_func(beamArcmin,noiseP,TCMB=TCMB)
Nlfuncdict['BB'] = cmb.get_noise_func(beamArcmin,noiseP,TCMB=TCMB)


print modLMap.shape
modLMap = fftshift(modLMap)
lxMap = fftshift(lxMap)
lyMap = fftshift(lyMap)
lx = fftshift(lx)
ly = fftshift(ly)


dlx = np.diff(lxMap,axis=1)[0,0]
dly = np.diff(lyMap,axis=0)[0,0]


Lmin = kellmin #10.
Lmax = kellmax #2000.
Ls = np.logspace(np.log10(Lmin),np.log10(Lmax),num_ells)
#Ls = np.linspace(Lmin,Lmax,num_ells)

lx1 = lyMap
ly1 = lxMap
lx1sq = lx1**2.
ly1sq = ly1**2.
l1sq = lx1sq+ly1sq
l1 = np.sqrt(l1sq)
ly2 = -ly1.copy()
ly2sq = ly2**2.


Als = {}
for polComb in polCombList:
    Als[polComb] = []

st = time.time()
for polComb in polCombList:
    XY = polComb
    X,Y = XY
    YY = Y+Y


    # do this correctly for each polcomb
    if halo:
        WXYl1 = WXY(XY,theory,Nlfuncdict,l1)
        WXYl1[l1<cmbellmin]=0.
        WXYl1[l1>cmbellmax]=0.
        if gradCut is not None:
            WXYl1[l1>gradCut]=0.
        
    phi_l1 = np.arctan2(lx1,ly1)    


    for L in Ls:

        lx2 = L - lx1
        lx2sq = lx2**2.
        l2sq = lx2sq+ly2sq
        l2 = np.sqrt(l2sq)
        Ll1 = L*lx1
        Ll2 = L*lx2

        if halo:
            WYl2 = WY(YY,theory,Nlfuncdict,l2) 
            WYl2[l2<cmbellmin]=0.
            WYl2[l2>cmbellmax]=0.

        phi_l2 = np.arctan2(lx2,ly2)    


        if polComb!='TT':
            cosDelta = np.cos(2.*(phi_l1-phi_l2))
            sinDelta = np.sin(2.*(phi_l1-phi_l2))
        else:
            cosDelta = None
            sinDelta = None

        if Y=='T':
            cfact = 1.
        elif Y=='E':
            cfact = cosDelta
        elif Y=='B':
            cfact = sinDelta
        else:
            raise ValueError

        f = fXY(XY,theory,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta)
            

        if halo:
            Falpha = Ll1*WXYl1*WYl2*cfact
        else:
            if (XY in ['TE','ET']):
                fS = fXY(XY,theory,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=sinDelta)
            else:
                fS = None
            
            Falpha = F(XY,f,fS,theory,Nlfuncdict,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta)
            Falpha[l1<cmbellmin]=0.
            Falpha[l1>cmbellmax]=0.
            Falpha[l2<cmbellmin]=0.
            Falpha[l2>cmbellmax]=0.
        
        integral = (Falpha*f).sum()*dlx*dly
        #Alinv = 2.*integral/((2.*np.pi)**2.)/L**2.
        Alinv = integral/((2.*np.pi)**2.)/L**2.
        Als[polComb].append(1./(Alinv))

for polComb in polCombList:
    Als[polComb] = np.array(Als[polComb])*Ls**2./4.
    print Als[polComb]
print time.time()-st," seconds."

# pl = io.Plotter(scaleY='log',scaleX='log')
# pl.add(Ls,Als*Ls**2./4.)
# pl.done("nltt.png")



ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    

pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

# CHECK THAT NORM MATCHES HU/OK
#for polComb,col in zip(polCombList,colorList):
col = None
for polComb in polCombList:

    try:
        huFile = '/home/msyriac/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_'+polComb.lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')
    except:
        huFile = '/home/msyriac/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_'+polComb[::-1].lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')


    pl.add(Ls,4.*Als[polComb]/2./np.pi,color=col)
    pl.add(huell,hunl,ls='--',color=col,label=polComb)

pl.legendOn(loc='lower left',labsize=10)
pl.done("testbin.png")
