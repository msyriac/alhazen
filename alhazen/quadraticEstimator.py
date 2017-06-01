from __future__ import division
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import orphics.analysis.flatMaps as fmaps 

from scipy.fftpack import fftshift,ifftshift,fftfreq
from scipy.interpolate import interp1d
from flipper.fft import fft,ifft

from orphics.tools.stats import timeit, bin2D
import alhazen.quadFunctions as qfuncs

import time
import cPickle as pickle

def fillLowEll(ells,cls,ellmin):
    # Fill low ells with the same value
    low_index = np.where(ells>ellmin)[0][0]
    lowest_ell = ells[low_index]
    lowest_val = cls[low_index]
    fill_ells = np.arange(2,lowest_ell,1)
    new_ells = np.append(fill_ells,ells[low_index:])
    fill_cls = np.array([lowest_val]*len(fill_ells))
    new_cls = np.append(fill_cls,cls[low_index:])

    return new_ells,new_cls


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


# class MLEstimator(Estimator):

#     def __init__(self,templateLiteMap,
#                  theorySpectraForFilters,
#                  theorySpectraForNorm=None,
#                  noiseX2dTEB=[None,None,None],
#                  noiseY2dTEB=[None,None,None],
#                  fmaskX2dTEB=[None,None,None],
#                  fmaskY2dTEB=[None,None,None],
#                  fmaskKappa=None,
#                  doCurl=False,
#                  TOnly=False,
#                  halo=False,
#                  gradCut=None,
#                  verbose=False,
#                  loadPickledNormAndFilters=None,
#                  savePickledNormAndFilters=None,
#                  numIterations=20):
    
#         self.Estimator(templateLiteMap,
#                  theorySpectraForFilters,
#                  theorySpectraForNorm,
#                  noiseX2dTEB,
#                  noiseY2dTEB,
#                  fmaskX2dTEB,
#                  fmaskY2dTEB,
#                  fmaskKappa,
#                  doCurl,
#                  TOnly,
#                  halo,
#                  gradCut,
#                  verbose,
#                  loadPickledNormAndFilters,
#                  savePickledNormAndFilters)

        
        
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
        #B = fft(self.modLMap,axes=[-2,-1],flags=['FFTW_MEASURE'])


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
        self.uClFid2d[XY] = power2dData.copy()+0.j
    def addUnlensedNorm2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the CMB normalization, and will
        be perturbed if/when calculating derivatives.
        '''
        self.uClNow2d[XY] = power2dData.copy()+0.j
    def addLensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.lClFid2d[XY] = power2dData.copy()+0.j
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
        self.noiseXX2d[XX] = power2dData.copy()+0.j
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
        self.noiseYY2d[YY] = power2dData.copy()+0.j
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
        self.clkk2d = power2dData.copy()+0.j
        self.clpp2d = 0.j+self.clkk2d.copy()*4./(self.modLMap**2.)/((self.modLMap+1.)**2.)


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
        raise NotImplementedError
    
    def getNlkk2d(self,XY,halo=False):
        if not(halo): raise NotImplementedError
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
                    

                    calc = ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True)+ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])
                    allTerms += [calc]
                    

            # else:

            #     clunlenTTArr = self.uClFid2d['TT'].copy()

            #     preG = self.WY('TT') #np.nan_to_num(1./cltotTTArrY)

            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr*np.nan_to_num(1./cltotTTArrX)/2.            
            #         preFX = ell1*clunlenTTArrNow*np.nan_to_num(1./cltotTTArrX)
            #         preGX = ell2*clunlenTTArr*np.nan_to_num(1./cltotTTArrY)


                    
            #         calc = 2.*ell1*ell2*fft(ifft(preF,axes=[-2,-1])*ifft(preG,axes=[-2,-1])+ifft(preFX,axes=[-2,-1])*ifft(preGX,axes=[-2,-1])/2.,axes=[-2,-1])
            #         allTerms += [calc]
          

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
                        allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                        
                        #allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                        
                        preFX = trigfact*ell1*clunlenEEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                        #allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                
            # else:


            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = trigfact*ell1*ell2*clunlenEEArrNow*clunlenEEArr*np.nan_to_num(1./cltotEEArr)/2.
            #             preG = trigfact*np.nan_to_num(1./cltotEEArr)
            #             preFX = trigfact*ell1*clunlenEEArrNow*np.nan_to_num(1./cltotEEArr)
            #             preGX = trigfact*ell2*clunlenEEArr*np.nan_to_num(1./cltotEEArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX)/2.)]


            


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
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)*ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]


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
                    allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    for trigfact in [cosf,sinf]:

                        preFX = trigfact*ell1*clunlenTEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]


            # else:



            #     sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            #     cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            #     lx = self.lxMap
            #     ly = self.lyMap

            
            #     lxhat = self.lxHatMap
            #     lyhat = self.lyHatMap

            #     sinf = sin2phi(lxhat,lyhat)
            #     sinsqf = sinf**2.
            #     cosf = cos2phi(lxhat,lyhat)
            #     cossqf = cosf**2.
                
                
            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         preF = ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
            #         preG = np.nan_to_num(1./cltotTTArr)
            #         allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = np.nan_to_num(1./cltotEEArr)
            #             preG = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotTTArr)
            #             allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cosf,sinf]:
                        
            #             preFX = trigfact*ell1*clunlenTEArrNow*np.nan_to_num(1./cltotEEArr)
            #             preGX = trigfact*ell2*clunlenTEArr*np.nan_to_num(1./cltotTTArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                    

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
                        allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]

                
            # else:



            #     sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            #     cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            #     lx = self.lxMap
            #     ly = self.lyMap

            
            #     lxhat = self.lxHatMap
            #     lyhat = self.lyHatMap

            #     sinf = sin2phi(lxhat,lyhat)
            #     sinsqf = sinf**2.
            #     cosf = cos2phi(lxhat,lyhat)
            #     cossqf = cosf**2.
                
                
            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = trigfact*ell1*ell2*clunlenTEArrNow* self.WXY('TE')#clunlenTEArr*np.nan_to_num(1./cltotTTArr)
            #             preG = trigfact*self.WY('EE')#np.nan_to_num(1./cltotEEArr)
            #             allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         preF = self.WY('TT')#np.nan_to_num(1./cltotTTArr)
            #         preG = ell1*ell2*clunlenTEArrNow* self.WXY('ET') #*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
            #         allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cosf,sinf]:
                        
            #             preFX = trigfact*ell1*clunlenTEArrNow*self.WY('TT')#np.nan_to_num(1./cltotTTArr)
            #             preGX = trigfact*ell2* self.WXY('ET')#*clunlenTEArr*np.nan_to_num(1./cltotEEArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]


                

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
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)*ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    

            


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
        clunlentotEEArr = (self.lClFid2d['EE'].copy()+self.noiseYY2d['EE'])
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

                allTerms += [trigfactOut*(fft(ifft(preF1,axes=[-2,-1],normalize=True)*ifft(preG1,axes=[-2,-1],normalize=True) - ifft(preF2,axes=[-2,-1],normalize=True)*ifft(preG2,axes=[-2,-1],normalize=True),axes=[-2,-1]))]


        
        ClBBres = np.real(np.sum( allTerms, axis = 0))

        
        ClBBres[np.where(np.logical_or(self.modLMap >= self.bigell, self.modLMap == 0.))] = 0.
        ClBBres *= self.Nx * self.Ny 
        ClBBres[self.fMaskYY['EE']==0] = 0.
                
        
        area =self.Nx*self.Ny*self.pixScaleX*self.pixScaleY
        bbNoise2D = ((np.sqrt(ClBBres)/self.pixScaleX/self.pixScaleY)**2.)*(area/(self.Nx*self.Ny*1.0)**2)

        self.lClFid2d['BB'] = bbNoise2D.copy()

        
        return bbNoise2D

                
    
def Nlmv(Nleach,pols,centers,nlkk,bin_edges):
    # Nleach: dict of (ls,Nls) for each polComb
    # pols: list of polCombs to include
    # centers,nlkk: additonal Nl to add
    
    Nlmvinv = 0.
    for polComb in pols:
        ls,Nls = Nleach[polComb]
        nlfunc = interp1d(ls,Nls,bounds_error=False,fill_value=np.inf)
        Nleval = nlfunc(bin_edges)
        Nlmvinv += np.nan_to_num(1./Nleval)
        
    if nlkk is not None:
        nlfunc = interp1d(centers,nlkk,bounds_error=False,fill_value=np.inf)
        Nleval = nlfunc(bin_edges)
        Nlmvinv += np.nan_to_num(1./Nleval)
        
    return np.nan_to_num(1./Nlmvinv)


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

    def updateNoiseAdvanced(self,beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesX,alphasX,lkneesY,alphasY,lxcutTX,lxcutTY,lycutTX,lycutTY,lxcutPX,lxcutPY,lycutPX,lycutPY,fgFuncX,fgFuncY,beamFileTX,beamFilePX,beamFileTY,beamFilePY,noiseFuncTX,noiseFuncTY,noiseFuncPX,noiseFuncPY):

        self.N.lmax_T = max(tellmaxX,tellmaxY)
        self.N.lmax_P = max(pellmaxX,pellmaxY)

        lkneeTX, lkneePX = lkneesX
        lkneeTY, lkneePY = lkneesY
        alphaTX, alphaPX = alphasX
        alphaTY, alphaPY = alphasY
        

        nTX = fmaps.whiteNoise2D([noiseTX],beamTX,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneeTX],alphas=[alphaTX],\
                                 beamFile=beamFileTX, \
                                 noiseFuncs = [noiseFuncTX])[0]
        nTY = fmaps.whiteNoise2D([noiseTY],beamTY,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneeTY],alphas=[alphaTY], \
                                 beamFile=beamFileTY, \
                                 noiseFuncs=[noiseFuncTY])[0]
        nPX = fmaps.whiteNoise2D([noisePX],beamPX,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneePX],alphas=[alphaPX],\
                                 beamFile=beamFilePX, \
                                 noiseFuncs = [noiseFuncPX])[0]
        nPY = fmaps.whiteNoise2D([noisePY],beamPY,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneePY],alphas=[alphaPY], \
                                 beamFile=beamFilePY, \
                                 noiseFuncs=[noiseFuncPY])[0]


        
        fMaskTX = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=tellminX,lmax=tellmaxX,lxcut=lxcutTX,lycut=lycutTX)
        fMaskTY = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=tellminY,lmax=tellmaxY,lxcut=lxcutTY,lycut=lycutTY)
        fMaskPX = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=pellminX,lmax=pellmaxX,lxcut=lxcutPX,lycut=lycutPX)
        fMaskPY = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=pellminY,lmax=pellmaxY,lxcut=lxcutPY,lycut=lycutPY)

        if fgFuncX is not None:
            fg2d = fgFuncX(self.N.modLMap) #/ self.TCMB**2.
            nTX += fg2d
        if fgFuncY is not None:
            fg2d = fgFuncY(self.N.modLMap) #/ self.TCMB**2.
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

        
    def updateNoise(self,beamX,noiseTX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamY=None,noiseTY=None,noisePY=None,tellminY=None,tellmaxY=None,pellminY=None,pellmaxY=None,lkneesX=[0.,0.],alphasX=[1.,1.],lkneesY=[0.,0.],alphasY=[1.,1.],lxcutTX=0,lxcutTY=0,lycutTX=0,lycutTY=0,lxcutPX=0,lxcutPY=0,lycutPX=0,lycutPY=0,fgFuncX=None,beamFileX=None,fgFuncY=None,beamFileY=None,noiseFuncTX=None,noiseFuncTY=None,noiseFuncPX=None,noiseFuncPY=None):

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
                                     noiseFuncs = [noiseFuncTX,noiseFuncPX])
        nTY,nPY = fmaps.whiteNoise2D([noiseTY,noisePY],beamY,self.N.modLMap, \
                                     TCMB=self.TCMB,lknees=lkneesY,alphas=alphasY,beamFile=beamFileY, \
                                     noiseFuncs=[noiseFuncTY,noiseFuncPY])


        
        fMaskTX = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=tellminX,lmax=tellmaxX,lxcut=lxcutTX,lycut=lycutTX)
        fMaskTY = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=tellminY,lmax=tellmaxY,lxcut=lxcutTY,lycut=lycutTY)
        fMaskPX = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=pellminX,lmax=pellmaxX,lxcut=lxcutPX,lycut=lycutPX)
        fMaskPY = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=pellminY,lmax=pellmaxY,lxcut=lxcutPY,lycut=lycutPY)

        if fgFuncX is not None:
            fg2d = fgFuncX(self.N.modLMap) #/ self.TCMB**2.
            nTX += fg2d
        if fgFuncY is not None:
            fg2d = fgFuncY(self.N.modLMap) #/ self.TCMB**2.
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

    def getNlIterative(self,polCombs,kmin,kmax,tellmax,pellmin,pellmax,dell=20,halo=True,dTolPercentage=1.,verbose=True,plot=False):
        
        Nleach = {}
        bin_edges = np.arange(kmin-dell/2.,kmax+dell/2.,dell)#+dell
        for polComb in polCombs:
            self.updateBins(bin_edges)
            AL = self.N.getNlkk2d(polComb,halo=halo)
            data2d = self.N.Nlkk[polComb]
            ls, Nls = self.binner.bin(data2d)
            Nls = sanitizePower(Nls)
            Nleach[polComb] = (ls,Nls)

        
        if ('EB' not in polCombs) and ('TB' not in polCombs):
            Nlret = Nlmv(Nleach,polCombs,None,None,bin_edges)
            return bin_edges,sanitizePower(Nlret),None,None,None

        origBB = self.N.lClFid2d['BB'].copy()
        delensBinner =  bin2D(self.N.modLMap, bin_edges)
        ellsOrig, oclbb = delensBinner.bin(origBB)
        oclbb = sanitizePower(oclbb)
        origclbb = oclbb.copy()

        if plot:
            from orphics.tools.io import Plotter
            pl = Plotter(scaleY='log',scaleX='log')
            pl.add(ellsOrig,oclbb*ellsOrig**2.,color='black',lw=2)
            
        ctol = np.inf
        inum = 0
        while ctol>dTolPercentage:
            bNlsinv = 0.
            polPass = list(polCombs)
            if verbose: print "Performing iteration ", inum+1
            for pol in ['EB','TB']:
                if not(pol in polCombs): continue
                Al2d = self.N.getNlkk2d(pol,halo)
                centers, nlkkeach = delensBinner.bin(self.N.Nlkk[pol])
                nlkkeach = sanitizePower(nlkkeach)
                bNlsinv += 1./nlkkeach
                polPass.remove(pol)
            nlkk = 1./bNlsinv
            
            Nldelens = Nlmv(Nleach,polPass,centers,nlkk,bin_edges)
            Nldelens2d = interp1d(bin_edges,Nldelens,fill_value=0.,bounds_error=False)(self.N.modLMap)

            bbNoise2D = self.N.delensClBB(Nldelens2d,halo)
            ells, dclbb = delensBinner.bin(bbNoise2D)
            dclbb = sanitizePower(dclbb)
            dclbb[ells<pellmin] = oclbb[ellsOrig<pellmin].copy()
            if inum>0:
                newLens = np.nanmean(nlkk)
                oldLens = np.nanmean(oldNl)
                new = np.nanmean(dclbb)
                old = np.nanmean(oclbb)
                ctol = np.abs(old-new)*100./new
                ctolLens = np.abs(oldLens-newLens)*100./newLens
                if verbose: print "Percentage difference between iterations is ",ctol, " compared to requested tolerance of ", dTolPercentage,". Diff of Nlkks is ",ctolLens
            oldNl = nlkk.copy()
            oclbb = dclbb.copy()
            inum += 1
            if plot:
                pl.add(ells,dclbb*ells**2.,ls="--",alpha=0.5,color="black")

        if plot:
            import os
            pl.done(os.environ['WWW']+'delens.png')
        self.N.lClFid2d['BB'] = origBB.copy()
        efficiency = ((origclbb-dclbb)*100./origclbb).max()


        new_ells,new_bb = fillLowEll(ells,dclbb,pellmin)
        new_k_ells,new_nlkk = fillLowEll(bin_edges,sanitizePower(Nldelens),kmin)
        
        return new_k_ells,new_nlkk,new_ells,new_bb,efficiency


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
        theorySpectraForFilters: an orphics.tools.cmb.TheorySpectra object with CMB Cls loaded
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
            self.kT = fft(T2DData,axes=[-2,-1])
        self.kGradx['T'] = lx*self.kT.copy()*1j
        self.kGrady['T'] = ly*self.kT.copy()*1j

        if E2DData is not None:
            if alreadyFTed:
                self.kE = E2DData
            else:
                self.kE = fft(E2DData,axes=[-2,-1])
            self.kGradx['E'] = 1.j*lx*self.kE.copy()
            self.kGrady['E'] = 1.j*ly*self.kE.copy()
        if B2DData is not None:
            if alreadyFTed:
                self.kB = B2DData
            else:
                self.kB = fft(B2DData,axes=[-2,-1])
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
                self.kHigh['T']=fft(T2DData,axes=[-2,-1])
        else:
            self.kHigh['T']=self.kT.copy()
        if E2DData is not None:
            if alreadyFTed:
                self.kHigh['E']=E2DData
            else:
                self.kHigh['E']=fft(E2DData,axes=[-2,-1])
        else:
            try:
                self.kHigh['E']=self.kE.copy()
            except:
                pass

        if B2DData is not None:
            if alreadyFTed:
                self.kHigh['B']=B2DData
            else:
                self.kHigh['B']=fft(B2DData,axes=[-2,-1])
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

        HighMapStar = ifft(self.kHigh[Y]*WY*phaseY*fMask*phaseB,axes=[-2,-1],normalize=True).conjugate()
        kPx = fft(ifft(self.kGradx[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])
        kPy = fft(ifft(self.kGrady[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])
        rawKappa = ifft(1.j*lx*kPx*fMask + 1.j*ly*kPy*fMask,axes=[-2,-1],normalize=True).real
        AL = np.nan_to_num(self.AL[XY])*fMask


        kappaft = -AL*fft(rawKappa,axes=[-2,-1])
        #if weightedFt: return np.nan_to_num(kappaft/self.N.Nlkk[XY]),np.nan_to_num(1./self.N.Nlkk[XY])
        self.kappa = ifft(kappaft,axes=[-2,-1],normalize=True)

        assert not(np.any(np.isnan(self.kappa)))
        # from orphics.tools.io import Plotter
        # pl = Plotter()
        # #pl.plot2d(np.nan_to_num(self.kappa))
        # pl.plot2d((self.kappa.real))
        # pl.done("output/nankappa.png")
        # sys.exit(0)
        # try:
        #     assert not(np.any(np.isnan(self.kappa)))
        # except:
        #     from orphics.tools.io import Plotter
        #     pl = Plotter()
        #     pl.plot2d(np.nan_to_num(self.kappa))
        #     pl.done("output/nankappa.png")
        #     sys.exit(0)

        if self.verbose:
            elapTime = time.time() - startTime
            print "Time for core kappa was ", elapTime ," seconds."

        if self.doCurl:
            OmAL = self.OmAL[XY]*fMask
            rawCurl = ifft(1.j*lx*kPy - 1.j*ly*kPx,axes=[-2,-1],normalize=True).real
            self.curl = -ifft(OmAL*fft(rawCurl,axes=[-2,-1]),axes=[-2,-1],normalize=True)
            return self.kappa, self.curl



            
        return self.kappa



def lensing_noise_including_fg(polCombList,theory,beamX,noiseTX,noisePX,lkneeTX,lkneePX,alphaTX,alphaPX,beamY,noiseTY,noisePY,lkneeTY,lkneePY,alphaTY,alphaPY,kellmin,kellmax,num_ells,independentExperiments=False,halo=True,gradCut=10000,fgFreqX=None,fgFreqY=None,constDict=None,ksz_battaglia_test_csv=None,tsz_battaglia_template_csv=None,degx = 5.,degy = 5.,px = 1.5,TCMB = 2.7255e6):

    if (fgFreqX is not None) or (fgFreqY is not None):
        from szlib.szcounts import fgNoises
        fgs = fgNoises(constDict,ksz_battaglia_test_csv,tsz_battaglia_template_csv)
        
    
    Ls,Nls,crosses,Nmv = isotropic_noise_full_lensing_covariance(polCombList,theory,noiseFuncTX,noiseFuncEX,noiseFuncBX,noiseFuncTY,noiseFuncEY,noiseFuncBY,kellmin,kellmax,num_ells,spacing="linear",independentExperiments=independentExperiments,degx = degx,degy = degy,px = px,TCMB = TCMB,halo=halo,gradCut=gradCut)

    
@timeit
def isotropic_noise_full_lensing_covariance(polCombList,theory,noiseFuncTX,noiseFuncEX,noiseFuncBX,noiseFuncTY,noiseFuncEY,noiseFuncBY,kellmin,kellmax,num_ells,spacing="linear",independentExperiments=False,degx = 5.,degy = 5.,px = 1.5,TCMB = 2.7255e6,halo=False,gradCut=10000,fgFuncTX=lambda x: 0.,fgFuncEX=lambda x: 0.,fgFuncBX=lambda x: 0.,fgFuncTY=lambda x: 0.,fgFuncEY=lambda x: 0.,fgFuncBY=lambda x: 0.):
    '''Quadratic estimator lensing minimum variance noise curves including full covariance.

    noiseFuncs are functions of ell for total beam-deconvolved noise including foregrounds. Set to inf beyond the ellranges of interest.

    
    '''
    
    import flipper.liteMap as lm
    import itertools
    hugeTemplate = lm.makeEmptyCEATemplate(degx,degy,pixScaleXarcmin=px,pixScaleYarcmin=px)
    lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(hugeTemplate)

    NlfuncdictX = {}
    NlfuncdictY = {}
    NlfuncdictX['TT'] = lambda x: noiseFuncTX(x)+fgFuncTX(x)
    NlfuncdictX['EE'] = lambda x: noiseFuncEX(x)+fgFuncEX(x)
    NlfuncdictX['BB'] = lambda x: noiseFuncBX(x)+fgFuncBX(x)
    NlfuncdictY['TT'] = lambda x: noiseFuncTY(x)+fgFuncTY(x)
    NlfuncdictY['EE'] = lambda x: noiseFuncEY(x)+fgFuncEY(x)
    NlfuncdictY['BB'] = lambda x: noiseFuncBY(x)+fgFuncBY(x)

    nfreq = modLMap.max()
    # assert nfreq>cmbellmax, "You need to make px smaller if you want to use a cmbellmax as high as "+str(cmbellmax)
    assert nfreq>kellmax, "You need to make px smaller if you want to use a kellmax as high as "+str(kellmax)

    dlx = np.diff(lxMap,axis=1)[0,0]
    dly = np.diff(lyMap,axis=0)[0,0]


    Lmin = kellmin
    Lmax = kellmax
    from orphics.tools.stats import npspace
    Ls = npspace(Lmin,Lmax,num_ells,spacing)
    
    lx1 = lyMap
    ly1 = lxMap
    lx1sq = lx1**2.
    ly1sq = ly1**2.
    l1sq = lx1sq+ly1sq
    l1 = np.sqrt(l1sq)
    ly2 = -ly1.copy()
    ly2sq = ly2**2.
    phi_l1 = np.arctan2(lx1,ly1)    


    crosses = {}
    polCrosses = itertools.combinations_with_replacement(polCombList,2)


    Cllist = ['TT','TE','EE','BB','ET']


    Als = {}
    for polComb in polCombList:
        Als[polComb] = []
    
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

            f = qfuncs.fXY(XY,theory,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta)


            if (XY in ['TE','ET']):
                fS = qfuncs.fXY(XY,theory,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=-sinDelta)
            else:
                fS = None

            Falpha = qfuncs.F(XY,f,fS,theory,NlfuncdictX,NlfuncdictY,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta,halo=halo,gradCut=gradCut)


            integral = (Falpha*f).sum()*dlx*dly
            Alinv = integral/((2.*np.pi)**2.)/L**2.
            Als[polComb].append(1./(Alinv))



    for polComb in polCombList:
        Als[polComb] = np.array(Als[polComb])
        

    for alpha,beta in polCrosses:
        #print alpha,beta
        Xalpha,Yalpha = alpha
        Xbeta,Ybeta = beta

        combs1 = [Xalpha+Xbeta,Yalpha+Ybeta]
        combs2 = [Xalpha+Ybeta,Yalpha+Xbeta]
        if not( all([combs in Cllist for combs in combs1])) and not(all([combs in Cllist for combs in combs2]) ):
            #print "skipping"
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


            falpha = qfuncs.fXY(alpha,theory,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta)
            fbeta = qfuncs.fXY(beta,theory,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta)


            falphaS = qfuncs.fXY(alpha,theory,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=-sinDelta)
            fbetaS = qfuncs.fXY(beta,theory,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=-sinDelta)

            Falpha = qfuncs.F(alpha,falpha,falphaS,theory,NlfuncdictX,NlfuncdictY,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta,halo=halo,gradCut=gradCut)
            Fbeta = qfuncs.F(beta,fbeta,fbetaS,theory,NlfuncdictX,NlfuncdictY,Ll1,Ll2,l1,l2,cos2phi=cosDelta,sin2phi=sinDelta,halo=halo,gradCut=gradCut)
            FbetaS = qfuncs.F(beta,fbetaS,fbeta,theory,NlfuncdictX,NlfuncdictY,Ll2,Ll1,l2,l1,cos2phi=cosDelta,sin2phi=-sinDelta,halo=halo,gradCut=gradCut)
            integral = qfuncs.crossIntegrand(alpha,beta,theory,NlfuncdictX,NlfuncdictY,Falpha,Fbeta,FbetaS,l1,l2,independentExperiments=independentExperiments).sum()*dlx*dly
            N = integral/((2.*np.pi)**2.)
            crosses[alpha+beta].append(N)
        crosses[alpha+beta] = Als[alpha]*Als[beta]*np.array(crosses[alpha+beta])/4.

    Nls = {}
    for polComb in polCombList:
        Nls[polComb] = Als[polComb]*Ls**2./4.

    print "Calculating mv..."
    Nmv = []
    for k,L in enumerate(Ls):
        Nmat = np.zeros((len(polCombList),len(polCombList)))
        polCrosses = itertools.combinations_with_replacement(polCombList,2)
        for alpha,beta in polCrosses:
            #print alpha,beta
            Xalpha,Yalpha = alpha
            Xbeta,Ybeta = beta

            combs1 = [Xalpha+Xbeta,Yalpha+Ybeta]
            combs2 = [Xalpha+Ybeta,Yalpha+Xbeta]
            if not( all([combs in Cllist for combs in combs1])) and not(all([combs in Cllist for combs in combs2]) ):
                #print "skipping"
                continue
            i = polCombList.index(alpha)
            j = polCombList.index(beta)
            Nmat[i,j] = crosses[alpha+beta][k]
            Nmat[j,i] = crosses[alpha+beta][k]
        Ninv = np.linalg.inv(Nmat)
        Nmv.append(1./Ninv.sum())
    Nmv = np.array(Nmv)
    return Ls,Nls,crosses,Nmv


def residualBB(Ls,Clkk,Nlkk,theory,noiseFuncEX,fgFuncEX):

    lx1 = lyMap
    ly1 = lxMap
    lx1sq = lx1**2.
    ly1sq = ly1**2.
    l1sq = lx1sq+ly1sq
    l1 = np.sqrt(l1sq)
    ly2 = -ly1.copy()
    ly2sq = ly2**2.
    phi_l1 = np.arctan2(lx1,ly1)    



class FullCov(object):

    def __init__(degx = 5.,degy = 5.,px = 1.5,TCMB = 2.7255e6):
        import flipper.liteMap as lm
        hugeTemplate = lm.makeEmptyCEATemplate(degx,degy,pixScaleXarcmin=px,pixScaleYarcmin=px)
        self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly = fmaps.getFTAttributesFromLiteMap(hugeTemplate)
