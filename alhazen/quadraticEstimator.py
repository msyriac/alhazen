from __future__ import division
import numpy as np
import orphics.analysis.flatMaps as fmaps 

'''
This module relies heavily on FFTs, so the keywords
fft2 and ifft2 are reserved for the chosen implementation.
'''
#from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq
#from numpy.fft import fft2,ifft2,fftshift,ifftshift,fftfreq
from scipy.fftpack import fftshift,ifftshift,fftfreq
from pyfftw.interfaces.scipy_fftpack import fft2
from pyfftw.interfaces.scipy_fftpack import ifft2

from orphics.tools.stats import timeit, bin2D

import time
import cPickle as pickle

def sanitizePower(Nlbinned):
    Nlbinned[Nlbinned<0.] = np.nan

    # fill nans with interp
    ok = -np.isnan(Nlbinned)
    xp = ok.ravel().nonzero()[0]
    fp = Nlbinned[-np.isnan(Nlbinned)]
    x  = np.isnan(Nlbinned).ravel().nonzero()[0]
    Nlbinned[np.isnan(Nlbinned)] = np.interp(x, xp, fp)
    return Nlbinned


def getMax(polComb,tellmax,pellmax):
    if polComb=='TT':
        return tellmax
    elif polComb in ['EE','EB']:
        return pellmax
    else:
        return max(tellmax,pellmax)


class QuadNorm(object):

    
    def __init__(self,templateMap,gradCut=None,verbose=False):
        '''

        templateFT is a template liteMap FFT object
    

    
        '''
        self.verbose = verbose
        self.Ny,self.Nx = templateMap.Ny, templateMap.Nx
        self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly = fmaps.getFTAttributesFromLiteMap(templateMap)
        self.lxHatMap = self.lxMap*np.nan_to_num(1. / self.modLMap)
        self.lyHatMap = self.lyMap*np.nan_to_num(1. / self.modLMap)


        self.uClNow2d = {}
        self.uClFid2d = {}
        self.lClFid2d = {}
        self.noiseXX2d = {}
        self.noiseYY2d = {}
        self.fMaskXX = {}
        self.fMaskYY = {}

        self.lmax_T=9000.
        self.lmax_P=9000.
        self.defaultMaskT = fmaps.fourierMask(self.lx,self.ly,self.modLMap,lmin=2,lmax=self.lmax_T)
        self.defaultMaskP = fmaps.fourierMask(self.lx,self.ly,self.modLMap,lmin=2,lmax=self.lmax_P)
        self.bigell=9000.
        if gradCut is not None: 
            self.gradCut = gradCut
        else:
            self.gradCut = bigell
        


        self.Nlkk = {}
        self.pixScaleX = templateMap.pixScaleX
        self.pixScaleY = templateMap.pixScaleY
        

    def __getstate__(self):
        # Clkk2d is not pickled yet!!!
        return self.verbose, self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly, self.lxHatMap, self.lyHatMap,self.uClNow2d, self.uClFid2d, self.lClFid2d, self.noiseXX2d, self.noiseYY2d, self.fMaskXX, self.fMaskYY, self.lmax_T, self.lmax_P, self.defaultMaskT, self.defaultMaskP, self.bigell, self.gradCut,self.Nlkk,self.pixScaleX,self.pixScaleY



    def __setstate__(self, state):
        self.verbose, self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly, self.lxHatMap, self.lyHatMap,self.uClNow2d, self.uClFid2d, self.lClFid2d, self.noiseXX2d, self.noiseYY2d, self.fMaskXX, self.fMaskYY, self.lmax_T, self.lmax_P, self.defaultMaskT, self.defaultMaskP, self.bigell, self.gradCut,self.Nlkk,self.pixScaleX,self.pixScaleY = state


    def addUnlensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.uClFid2d[XY] = power2dData.copy()
    def addUnlensedNorm2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the CMB normalization, and will
        be perturbed if/when calculating derivatives.
        '''
        self.uClNow2d[XY] = power2dData.copy()
    def addLensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.lClFid2d[XY] = power2dData.copy()
    def addNoise2DPowerXX(self,XX,power2dData,fourierMask=None):
        '''
        Noise power for the X leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        # check if fourier mask is int!
        self.noiseXX2d[XX] = power2dData.copy()
        if fourierMask is not None:
            self.noiseXX2d[XX][fourierMask==0] = np.inf
            self.fMaskXX[XX] = fourierMask
        else:
            if XX=='TT':
                self.noiseXX2d[XX][self.defaultMaskT==0] = np.inf
            else:
                self.noiseXX2d[XX][self.defaultMaskP==0] = np.inf

    def addNoise2DPowerYY(self,YY,power2dData,fourierMask=None):
        '''
        Noise power for the Y leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        # check if fourier mask is int!
        self.noiseYY2d[YY] = power2dData.copy()
        if fourierMask is not None:
            self.noiseYY2d[YY][fourierMask==0] = np.inf
            self.fMaskYY[YY] = fourierMask
        else:
            if YY=='TT':
                self.noiseYY2d[YY][self.defaultMaskT==0] = np.inf
            else:
                self.noiseYY2d[YY][self.defaultMaskP==0] = np.inf
        
    def addClkk2DPower(self,power2dData):
        '''
        Fiducial Clkk power
        Used if delensing
        power2d is a flipper power2d object            
        '''
        self.clkk2d = power2dData.copy()
        self.clpp2d = self.clkk2d.copy()*4./(self.modLMap**2.)/((self.modLMap+1.)**2.)


    def WXY(self,XY):
        X,Y = XY
        if Y=='B': Y='E'
        gradClXY = X+Y
        if XY=='ET': gradClXY = 'TE'
        W = np.nan_to_num(self.uClFid2d[gradClXY].copy()/(self.lClFid2d[X+X].copy()+self.noiseXX2d[X+X].copy()))*self.fMaskXX[X+X]
        W[self.modLMap>self.gradCut]=0.
        if X=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.


        return W
        

    def WY(self,YY):
        assert YY[0]==YY[1]
        W = np.nan_to_num(1./(self.lClFid2d[YY].copy()+self.noiseYY2d[YY].copy()))*self.fMaskYY[YY]
        W[np.where(self.modLMap >= self.lmax_T)] = 0.
        if YY[0]=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.
        return W

    def getCurlNlkk2d(self,XY,halo=False):
        pass
            
    def getNlkk2d(self,XY,halo=False):
        lx,ly = self.lxMap,self.lyMap
        lmap = self.modLMap

        if self.verbose: 
            print "Calculating norm for ", XY

            
        h=0.

        allTerms = []
            
        if XY == 'TT':
            
            clunlenTTArrNow = self.uClNow2d['TT'].copy()
                

            if halo:
            
                WXY = self.WXY('TT')
                WY = self.WY('TT')

                # binrange = np.arange(50,4000,10)
                # binner = bin2D(self.modLMap,binrange)
                # self.cents,self.wxy = binner.bin(WXY)

                preG = WY
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*WXY
                    preFX = ell1*WXY
                    preGX = ell2*clunlenTTArrNow*WY
                    

                    calc = ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX))
                    allTerms += [calc]
                    

            else:

                clunlenTTArr = self.uClFid2d['TT'].copy()

                preG = self.WY('TT') #np.nan_to_num(1./cltotTTArrY)

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr*np.nan_to_num(1./cltotTTArrX)/2.            
                    preFX = ell1*clunlenTTArrNow*np.nan_to_num(1./cltotTTArrX)
                    preGX = ell2*clunlenTTArr*np.nan_to_num(1./cltotTTArrY)


                    
                    calc = 2.*ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX)/2.)
                    allTerms += [calc]
          

        elif XY == 'EE':

            clunlenEEArrNow = self.uClNow2d['EE'].copy()


            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap


            lxhat = self.lxHatMap
            lyhat = self.lyHatMap

            sinf = sin2phi(lxhat,lyhat)
            sinsqf = sinf**2.
            cosf = cos2phi(lxhat,lyhat)
            cossqf = cosf**2.
                                
            if halo:
            

                WXY = self.WXY('EE')
                WY = self.WY('EE')
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenEEArrNow*WXY
                        preG = trigfact*WY
                        allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                        
                        preFX = trigfact*ell1*clunlenEEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                
            else:


                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenEEArrNow*clunlenEEArr*np.nan_to_num(1./cltotEEArr)/2.
                        preG = trigfact*np.nan_to_num(1./cltotEEArr)
                        preFX = trigfact*ell1*clunlenEEArrNow*np.nan_to_num(1./cltotEEArr)
                        preGX = trigfact*ell2*clunlenEEArr*np.nan_to_num(1./cltotEEArr)

                        allTerms += [2.*ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX)/2.)]


            


        elif XY == 'EB':


            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenBBArrNow = self.uClNow2d['BB'].copy()


            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap

            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            lxhat = self.lxHatMap
            lyhat = self.lyHatMap

            WXY = self.WXY('EB')
            WY = self.WY('BB')
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenEEArrNow*WXY
                preG = WY

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft2(ifft2(termF(preF,lxhat,lyhat))*ifft2(termG(preG,lxhat,lyhat)))]


        elif XY=='ET':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()

            if halo:
                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap


                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.


                WXY = self.WXY('ET')
                WY = self.WY('TT')

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTEArrNow*WXY
                    preG = WY
                    allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cosf,sinf]:

                        preFX = trigfact*ell1*clunlenTEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]


            else:



                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap

            
                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.
                
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
                    preG = np.nan_to_num(1./cltotTTArr)
                    allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = np.nan_to_num(1./cltotEEArr)
                        preG = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotTTArr)
                        allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*np.nan_to_num(1./cltotEEArr)
                        preGX = trigfact*ell2*clunlenTEArr*np.nan_to_num(1./cltotTTArr)

                        allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                    

        elif XY=='TE':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()

            if halo:
            
                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap

            
                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.
                
                WXY = self.WXY('TE')
                WY = self.WY('EE')

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenTEArrNow*WXY
                        preG = trigfact*WY
                        allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                
            else:



                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap

            
                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.
                
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenTEArrNow* self.WXY('TE')#clunlenTEArr*np.nan_to_num(1./cltotTTArr)
                        preG = trigfact*self.WY('EE')#np.nan_to_num(1./cltotEEArr)
                        allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    preF = self.WY('TT')#np.nan_to_num(1./cltotTTArr)
                    preG = ell1*ell2*clunlenTEArrNow* self.WXY('ET') #*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
                    allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*self.WY('TT')#np.nan_to_num(1./cltotTTArr)
                        preGX = trigfact*ell2* self.WXY('ET')#*clunlenTEArr*np.nan_to_num(1./cltotEEArr)

                        allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]


                

        elif XY == 'TB':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()


                
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap

            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            
            lxhat = self.lxHatMap
            lyhat = self.lyHatMap
            
            WXY = self.WXY('TB')
            WY = self.WY('BB')
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenTEArrNow*WXY
                preG = WY

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft2(ifft2(termF(preF,lxhat,lyhat))*ifft2(termG(preG,lxhat,lyhat)))]
                    

            


        else:
            print "ERROR: Unrecognized polComb"
            sys.exit(1)    
        
                        
        ALinv = np.real(np.sum( allTerms, axis = 0))
        NL = (lmap**2.) * ((lmap + 1.)**2.) *np.nan_to_num(1. / ALinv)/ 4.
        NL[np.where(np.logical_or(lmap >= self.bigell, lmap == 0.))] = 0.

        retval = np.nan_to_num(NL.real * self.pixScaleX*self.pixScaleY  )

        self.Nlkk[XY] = retval.copy()



        
        return retval * 2. * np.nan_to_num(1. / lmap/(lmap+1.))
        
        
                  

        
      


    def delensClBB(self,Nlkk,halo=True):
        self.Nlppnow = Nlkk*4./(self.modLMap**2.)/((self.modLMap+1.)**2.)
        clPPArr = self.clpp2d
        cltotPPArr = clPPArr + self.Nlppnow
        cltotPPArr[np.isnan(cltotPPArr)] = np.inf
        
        clunlenEEArr = self.uClFid2d['EE'].copy()
        clunlentotEEArr = (self.uClFid2d['EE'].copy()+self.noiseYY2d['EE'])
        clunlentotEEArr[self.fMaskYY['EE']==0] = np.inf
        clunlenEEArr[self.fMaskYY['EE']==0] = 0.
        clPPArr[self.fMaskYY['EE']==0] = 0.
        cltotPPArr[self.fMaskYY['EE']==0] = np.inf
        

        #if halo: clunlenEEArr[np.where(self.modLMap >= self.gradCut)] = 0.
                
        sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
        cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

        lx = self.lxMap
        ly = self.lyMap

            
        lxhat = self.lxHatMap
        lyhat = self.lyHatMap

        sinf = sin2phi(lxhat,lyhat)
        sinsqf = sinf**2.
        cosf = cos2phi(lxhat,lyhat)
        cossqf = cosf**2.

        
        allTerms = []
        for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
            for trigfactOut,trigfactIn in zip([sinsqf,cossqf,1.j*np.sqrt(2.)*sinf*cosf],[cossqf,sinsqf,1.j*np.sqrt(2.)*sinf*cosf]):
                preF1 = trigfactIn*ellsq*clunlenEEArr
                preG1 = ellsq*clPPArr

                preF2 = trigfactIn*ellsq*clunlenEEArr**2.*np.nan_to_num(1./clunlentotEEArr)
                preG2 = ellsq*clPPArr**2.*np.nan_to_num(1./cltotPPArr)

                allTerms += [trigfactOut*(fft2(ifft2(preF1)*ifft2(preG1) - ifft2(preF2)*ifft2(preG2)))]


        
        ClBBres = np.real(np.sum( allTerms, axis = 0))

        
        ClBBres[np.where(np.logical_or(self.modLMap >= self.bigell, self.modLMap == 0.))] = 0.
        ClBBres *= self.Nx * self.Ny 
        ClBBres[self.fMaskYY['EE']==0] = 0.
        from orphics.tools.io import Plotter
                
        
        area =self.Nx*self.Ny*self.pixScaleX*self.pixScaleY
        bbNoise2D = ((np.sqrt(ClBBres)/self.pixScaleX/self.pixScaleY)**2.)*(area/(self.Nx*self.Ny*1.0)**2)

        self.lClFid2d['BB'] = bbNoise2D.copy()

        
        return bbNoise2D

                
    


class NlGenerator(object):
    def __init__(self,templateMap,theorySpectra,bin_edges=None,gradCut=None,TCMB=2.725e6):

        self.N = QuadNorm(templateMap,gradCut=gradCut)
        self.TCMB = TCMB

        cmbList = ['TT','TE','EE','BB']
        
        
        for cmb in cmbList:
            uClFilt = theorySpectra.uCl(cmb,self.N.modLMap)
            uClNorm = uClFilt
            lClFilt = theorySpectra.lCl(cmb,self.N.modLMap)
            self.N.addUnlensedFilter2DPower(cmb,uClFilt)
            self.N.addLensedFilter2DPower(cmb,lClFilt)
            self.N.addUnlensedNorm2DPower(cmb,uClNorm)

        Clkk2d = theorySpectra.gCl("kk",self.N.modLMap)    
        self.N.addClkk2DPower(Clkk2d)
            

        if bin_edges is not None:
            self.bin_edges = bin_edges
            self.binner = bin2D(self.N.modLMap, bin_edges)

    def updateBins(self,bin_edges):
        self.N.bigell = bin_edges[len(bin_edges)-1]
        self.binner = bin2D(self.N.modLMap, bin_edges)
        self.bin_edges = bin_edges

    def updateNoise(self,beamX,noiseTX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamY=None,noiseTY=None,noisePY=None,tellminY=None,tellmaxY=None,pellminY=None,pellmaxY=None,lkneesX=[0.,0.],alphasX=[1.,1.],lkneesY=[0.,0.],alphasY=[1.,1.],lxcutTX=0,lxcutTY=0,lycutTX=0,lycutTY=0,lxcutPX=0,lxcutPY=0,lycutPX=0,lycutPY=0,fgFileX=None,beamFileX=None,fgFileY=None,beamFileY=None,noiseFileTX=None,noiseFileTY=None,noiseFilePX=None,noiseFilePY=None):

        def setDefault(A,B):
            if A is None:
                return B
            else:
                return A

        beamY = setDefault(beamY,beamX)
        noiseTY = setDefault(noiseTY,noiseTX)
        noisePY = setDefault(noisePY,noisePX)
        tellminY = setDefault(tellminY,tellminX)
        pellminY = setDefault(pellminY,pellminX)
        tellmaxY = setDefault(tellmaxY,tellmaxX)
        pellmaxY = setDefault(pellmaxY,pellmaxX)

        self.N.lmax_T = max(tellmaxX,tellmaxY)
        self.N.lmax_P = max(pellmaxX,pellmaxY)

        nTX,nPX = fmaps.whiteNoise2D([noiseTX,noisePX],beamX,self.N.modLMap, \
                                     TCMB=self.TCMB,lknees=lkneesX,alphas=alphasX,beamFile=beamFileX, \
                                     noiseFiles = [noiseFileTX,noiseFilePX])
        nTY,nPY = fmaps.whiteNoise2D([noiseTY,noisePY],beamY,self.N.modLMap, \
                                     TCMB=self.TCMB,lknees=lkneesY,alphas=alphasY,beamFile=beamFileY, \
                                     noiseFiles=[noiseFileTY,noiseFilePY])
        fMaskTX = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=tellminX,lmax=tellmaxX,lxcut=lxcutTX,lycut=lycutTX)
        fMaskTY = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=tellminY,lmax=tellmaxY,lxcut=lxcutTY,lycut=lycutTY)
        fMaskPX = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=pellminX,lmax=pellmaxX,lxcut=lxcutPX,lycut=lycutPX)
        fMaskPY = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=pellminY,lmax=pellmaxY,lxcut=lxcutPY,lycut=lycutPY)

        if fgFileX is not None:
            from scipy.interpolate import interp1d
            fells,fg = np.loadtxt(fgFileX,unpack=True)
            fgfunc = interp1d(fells,fg,bounds_error=False,fill_value=0.)
            fg2d = fgfunc(self.N.modLMap) / self.TCMB**2.
            nTX += fg2d
        if fgFileY is not None:
            from scipy.interpolate import interp1d
            fells,fg = np.loadtxt(fgFileY,unpack=True)
            fgfunc = interp1d(fells,fg,bounds_error=False,fill_value=0.)
            fg2d = fgfunc(self.N.modLMap) / self.TCMB**2.
            nTY += fg2d

            
        nList = ['TT','EE','BB']

        nListX = [nTX,nPX,nPX]
        nListY = [nTY,nPY,nPY]
        fListX = [fMaskTX,fMaskPX,fMaskPX]
        fListY = [fMaskTY,fMaskPY,fMaskPY]
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,nListX[i],fListX[i])
            self.N.addNoise2DPowerYY(noise,nListY[i],fListY[i])

        return nTX,nPX,nTY,nPY

    #@timeit
    def getNl(self,polComb='TT',halo=True):            

        AL = self.N.getNlkk2d(polComb,halo=halo)
        data2d = self.N.Nlkk[polComb]

        centers, Nlbinned = self.binner.bin(data2d)
        Nlbinned = sanitizePower(Nlbinned)
        
        return centers, Nlbinned

    def iterativeDelens(self,xy,dTolPercentage=1.0,halo=True,verbose=True):
        assert xy=='EB' or xy=='TB'
        origBB = self.N.lClFid2d['BB'].copy()
        bin_edges = self.bin_edges #np.arange(100.,3000.,20.)
        delensBinner =  bin2D(self.N.modLMap, bin_edges)
        ells, oclbb = delensBinner.bin(origBB)
        oclbb = sanitizePower(oclbb)

        ctol = np.inf
        inum = 0


        
        #from orphics.tools.output import Plotter
        #pl = Plotter(scaleY='log',scaleX='log')
        #pl = Plotter(scaleY='log')
        while ctol>dTolPercentage:
            if verbose: print "Performing iteration ", inum+1
            Al2d = self.N.getNlkk2d(xy,halo)
            centers, nlkk = delensBinner.bin(self.N.Nlkk[xy])
            nlkk = sanitizePower(nlkk)
            bbNoise2D = self.N.delensClBB(self.N.Nlkk[xy],halo)
            ells, dclbb = delensBinner.bin(bbNoise2D)
            dclbb = sanitizePower(dclbb)
            if inum>0:
                new = np.nanmean(nlkk)
                old = np.nanmean(oldNl)
                ctol = np.abs(old-new)*100./new
                if verbose: print "Percentage difference between iterations is ",ctol, " compared to requested tolerance of ", dTolPercentage
            oldNl = nlkk.copy()
            inum += 1
            #pl.add(centers,nlkk)
            #pl.add(ells,dclbb*ells**2.)
        #pl.done('output/delens'+xy+'.png')
        self.N.lClFid2d['BB'] = origBB.copy()
        efficiency = (np.max(oclbb)-np.max(dclbb))*100./np.max(oclbb)
        return centers,nlkk,efficiency

class Estimator(object):
    '''
    Flat-sky lensing and Omega quadratic estimators
    Functionality includes:
    - small-scale lens estimation with gradient cutoff
    - combine maps from two different experiments


    NOTE: The TE estimator is not identical between large
    and small-scale estimators. Need to test this.
    '''


    def __init__(self,templateLiteMap,
                 theorySpectraForFilters,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[None,None,None],
                 noiseY2dTEB=[None,None,None],
                 fmaskX2dTEB=[None,None,None],
                 fmaskY2dTEB=[None,None,None],
                 fmaskKappa=None,
                 doCurl=False,
                 TOnly=False,
                 halo=False,
                 gradCut=None,
                 verbose=False,
                 loadPickledNormAndFilters=None,
                 savePickledNormAndFilters=None):

        '''
        All the 2d fourier objects below are pre-fftshifting. They must be of the same dimension.

        templateLiteMap: any object that contains the attributes Nx, Ny, pixScaleX, pixScaleY specifying map dimensions
        theorySpectraForFilters: a orphics.theory.gaussianCov.TheorySpectra object with CMB Cls loaded
        theorySpectraForNorm=None: same as above but if you want to use a different cosmology in the expected value of the 2-pt
        noiseX2dTEB=[None,None,None]: a list of 2d arrays that corresponds to the noise power in T, E, B (same units as Cls above)
        noiseY2dTEB=[None,None,None]: the same as above but if you want to use a different experiment for the Y maps
        fmaskX2dTEB=[None,None,None]: a list of 2d integer arrays where 1 corresponds to modes included and 0 to those not included
        fmaskY2dTEB=[None,None,None]: same as above but for Y maps
        fmaskKappa=None: same as above but for output kappa map
        doCurl=False: return curl Omega estimates too? If yes, output of getKappa will be (kappa,curl)
        TOnly=False: do only TT? If yes, others will not be initialized and you'll get errors if you try to getKappa(XY) for XY!=TT
        halo=False: use the halo lensing estimators?
        gradCut=None: if using halo lensing estimators, specify an integer up to what L the X map will be retained
        verbose=False: print some occasional output?

        '''

        self.verbose = verbose

        # initialize norm and filters

        self.doCurl = doCurl



        if loadPickledNormAndFilters is not None:
            if verbose: print "Unpickling..."
            with open(loadPickledNormAndFilters,'rb') as fin:
                self.N,self.AL,self.OmAL,self.fmaskK,self.phaseY = pickle.load(fin)
            return



        self.halo = halo
        self.AL = {}
        if doCurl: self.OmAL = {}



        self.N = QuadNorm(templateLiteMap,gradCut=gradCut,verbose=verbose)
        if fmaskKappa is None:
            ellMinK = 80
            ellMaxK = 3000
            print "WARNING: using default kappa mask of 80 < L < 3000"
            self.fmaskK = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=ellMinK,lmax=ellMaxK)
        else:
            self.fmaskK = fmaskKappa

        if TOnly: 
            nList = ['TT']
            cmbList = ['TT']
            estList = ['TT']
            self.phaseY = 1.
        else:
            self.phaseY = np.cos(2.*self.N.thetaMap)+1.j*np.sin(2.*self.N.thetaMap)
            nList = ['TT','EE','BB']
            cmbList = ['TT','TE','EE','BB']
            estList = ['TT','TE','ET','EB','EE','TB']

        
        if self.verbose: print "Initializing filters and normalization for quadratic estimators..."
        for cmb in cmbList:
            uClFilt = theorySpectraForFilters.uCl(cmb,self.N.modLMap)

            if theorySpectraForNorm is not None:
                uClNorm = theorySpectraForNorm.uCl(cmb,self.N.modLMap)
            else:
                uClNorm = uClFilt
            lClFilt = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
            self.N.addUnlensedFilter2DPower(cmb,uClFilt)
            self.N.addLensedFilter2DPower(cmb,lClFilt)
            self.N.addUnlensedNorm2DPower(cmb,uClNorm)
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,noiseX2dTEB[i],fmaskX2dTEB[i])
            self.N.addNoise2DPowerYY(noise,noiseY2dTEB[i],fmaskY2dTEB[i])


        self.OmAL = None
        for est in estList:
            self.AL[est] = self.N.getNlkk2d(est,halo=halo)
            if doCurl: self.OmAL[est] = self.N.getCurlNlkk2d(est,halo=halo)


        if savePickledNormAndFilters is not None:

            if verbose: print "Pickling..."
            with open(savePickledNormAndFilters,'wb') as fout:
                pickle.dump((self.N,self.AL,self.OmAL,self.fmaskK,self.phaseY),fout)


        

    def updateTEB_X(self,T2DData,E2DData=None,B2DData=None,alreadyFTed=False):
        '''
        Masking and windowing and apodizing and beam deconvolution has to be done beforehand!

        Maps must have units corresponding to those of theory Cls and noise power
        '''
        self._hasX = True

        self.kGradx = {}
        self.kGrady = {}

        lx = self.N.lxMap
        ly = self.N.lyMap

        if alreadyFTed:
            self.kT = T2DData
        else:
            self.kT = fft2(T2DData)
        self.kGradx['T'] = lx*self.kT.copy()*1j
        self.kGrady['T'] = ly*self.kT.copy()*1j

        if E2DData is not None:
            if alreadyFTed:
                self.kE = E2DData
            else:
                self.kE = fft2(E2DData)
            self.kGradx['E'] = 1.j*lx*self.kE.copy()
            self.kGrady['E'] = 1.j*ly*self.kE.copy()
        if B2DData is not None:
            if alreadyFTed:
                self.kB = B2DData
            else:
                self.kB = fft2(B2DData)
            self.kGradx['B'] = 1.j*lx*self.kB.copy()
            self.kGrady['B'] = 1.j*ly*self.kB.copy()
        
        

    def updateTEB_Y(self,T2DData=None,E2DData=None,B2DData=None,alreadyFTed=False):
        assert self._hasX, "Need to initialize gradient first."
        self._hasY = True
        
        self.kHigh = {}

        if T2DData is not None:
            if alreadyFTed:
                self.kHigh['T']=T2DData
            else:
                self.kHigh['T']=fft2(T2DData)
        else:
            self.kHigh['T']=self.kT.copy()
        if E2DData is not None:
            if alreadyFTed:
                self.kHigh['E']=E2DData
            else:
                self.kHigh['E']=fft2(E2DData)
        else:
            try:
                self.kHigh['E']=self.kE.copy()
            except:
                pass

        if B2DData is not None:
            if alreadyFTed:
                self.kHigh['B']=B2DData
            else:
                self.kHigh['B']=fft2(B2DData)
        else:
            try:
                self.kHigh['B']=self.kB.copy()
            except:
                pass

    def getKappa(self,XY,weightedFt=False):

        assert self._hasX and self._hasY
        assert XY in ['TT','TE','ET','EB','TB','EE']
        X,Y = XY

        WXY = self.N.WXY(XY)
        WY = self.N.WY(Y+Y)



        lx = self.N.lxMap
        ly = self.N.lyMap

        if Y in ['E','B']:
            phaseY = self.phaseY
        else:
            phaseY = 1.

        phaseB = (int(Y=='B')*1.j)+(int(Y!='B'))
        
        fMask = self.fmaskK

        if self.verbose: startTime = time.time()

        HighMapStar = ifft2(self.kHigh[Y]*WY*phaseY*fMask*phaseB).conjugate()
        kPx = fft2(ifft2(self.kGradx[X]*WXY*phaseY)*HighMapStar)
        kPy = fft2(ifft2(self.kGrady[X]*WXY*phaseY)*HighMapStar)
        rawKappa = ifft2(1.j*lx*kPx*fMask + 1.j*ly*kPy*fMask).real
        AL = self.AL[XY]*fMask


        kappaft = -AL*fft2(rawKappa)
        #if weightedFt: return np.nan_to_num(kappaft/self.N.Nlkk[XY]),np.nan_to_num(1./self.N.Nlkk[XY])
        self.kappa = ifft2(kappaft)


        if self.verbose:
            elapTime = time.time() - startTime
            print "Time for core kappa was ", elapTime ," seconds."

        if self.doCurl:
            OmAL = self.OmAL[XY]*fMask
            rawCurl = ifft2(1.j*lx*kPy - 1.j*ly*kPx).real
            self.curl = -ifft2(OmAL*fft2(rawCurl))
            return self.kappa, self.curl



            
        return self.kappa





