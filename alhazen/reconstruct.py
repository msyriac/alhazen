from __future__ import division
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import orphics.analysis.flatMaps as fmaps 

from scipy.fftpack import fftshift,ifftshift,fftfreq
from scipy.interpolate import interp1d
#from flipper.fft import fft,ifft
from enlib import enmap
from enlib.fft import fft,ifft

from orphics.tools.stats import timeit, bin2D
import alhazen.quadFunctions as qfuncs

import time
import cPickle as pickle

class QuadNormSmooth(object):

    
    def __init__(self,shape,wcs,gradCut=None,kBeamX=None,kBeamY=None):
        '''

        templateFT is a template liteMap FFT object
    

    
        '''
        self.Ny,self.Nx = shape[-2:]
        self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly = fmaps.get_ft_attributes_enmap(shape,wcs)
        self.lxHatMap = self.lxMap*np.nan_to_num(1. / self.modLMap)
        self.lyHatMap = self.lyMap*np.nan_to_num(1. / self.modLMap)

        if kBeamX is not None:           
            self.kBeamX = kBeamX
        else:
            self.kBeamX = 1.
            
        if kBeamY is not None:           
            self.kBeamY = kBeamY
        else:
            self.kBeamY = 1.


        self.uClNow2d = {}
        self.uClFid2d = {}
        self.lClFid2d = {}
        self.noiseXX2d = {}
        self.noiseYY2d = {}

        if gradCut is not None: 
            self.gradCut = gradCut
        else:
            self.gradCut = self.modLMap.max()
        


        self.Nlkk = {}
        self.pixScaleY, self.pixScaleX = enmap.pixshape(shape,wcs)
        


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
        
    def addNoise2DPowerXX(self,XX,power2dData):
        '''
        Noise power for the X leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        '''
        self.noiseXX2d[XX] = power2dData.copy()+0.j

    def addNoise2DPowerYY(self,YY,power2dData):
        '''
        Noise power for the Y leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        '''
        self.noiseYY2d[YY] = power2dData.copy()+0.j
        
    def addClkk2DPower(self,power2dData):
        '''
        Fiducial Clkk power
        Used if delensing
        power2d is a flipper power2d object            
        '''
        self.clkk2d = power2dData.copy()+0.j
        self.clpp2d = np.nan_to_num(0.j+self.clkk2d.copy()*4./(self.modLMap**2.)/((self.modLMap+1.)**2.))


    def WXY(self,XY):
        X,Y = XY
        if Y=='B': Y='E'
        gradClXY = X+Y
        if XY=='ET': gradClXY = 'TE'
        W = np.nan_to_num(self.uClFid2d[gradClXY].copy()/(self.lClFid2d[X+X].copy()+self.noiseXX2d[X+X].copy()))*self.kBeamX
        W[self.modLMap>self.gradCut]=0.


        return W
        

    def WY(self,YY):
        assert YY[0]==YY[1]
        W = np.nan_to_num(1./(self.lClFid2d[YY].copy()+self.noiseYY2d[YY].copy()))*self.kBeamY
        return W

    def getCurlNlkk2d(self,XY,halo=False):
        raise NotImplementedError
    
    def getNlkk2d(self,XY):
        lx,ly = self.lxMap,self.lyMap
        lmap = self.modLMap


            
        h=0.

        allTerms = []
            
        if XY == 'TT':
            
            clunlenTTArrNow = self.uClNow2d['TT'].copy()
                

            WXY = self.WXY('TT')*self.kBeamX
            WY = self.WY('TT')*self.kBeamY


            preG = WY
            rfact = 2.**0.25
            for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                preF = ell1*ell2*clunlenTTArrNow*WXY
                preFX = ell1*WXY
                preGX = ell2*clunlenTTArrNow*WY


                calc = ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True)+ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])
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
                                


            WXY = self.WXY('EE')*self.kBeamX
            WY = self.WY('EE')*self.kBeamY
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

            WXY = self.WXY('EB')*self.kBeamX
            WY = self.WY('BB')*self.kBeamY
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenEEArrNow*WXY
                preG = WY

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)*ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]


        elif XY=='ET':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()
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


            WXY = self.WXY('ET')*self.kBeamX
            WY = self.WY('TT')*self.kBeamY

            rfact = 2.**0.25
            for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                preF = ell1*ell2*clunlenTEArrNow*WXY
                preG = WY
                allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                for trigfact in [cosf,sinf]:

                    preFX = trigfact*ell1*clunlenTEArrNow*WY
                    preGX = trigfact*ell2*WXY

                    allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]


                    

        elif XY=='TE':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()


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

            WXY = self.WXY('TE')*self.kBeamX
            WY = self.WY('EE')*self.kBeamY

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
            
            WXY = self.WXY('TB')*self.kBeamX
            WY = self.WY('BB')*self.kBeamY
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
        NL[np.where(lmap < 2.)] = 0.

        retval = np.nan_to_num(NL.real * self.pixScaleX*self.pixScaleY  )

        self.Nlkk[XY] = retval.copy()



        
        return retval * 2. * np.nan_to_num(1. / lmap/(lmap+1.))
        
        
                  

class EstimatorSmooth(object):
    '''
    Flat-sky lensing and Omega quadratic estimators
    Functionality includes:
    - small-scale lens estimation with gradient cutoff
    - combine maps from two different experiments


    NOTE: The TE estimator is not identical between large
    and small-scale estimators. Need to test this.
    '''


    def __init__(self,shape,wcs,
                 theory_filters,
                 theory_norm=None,
                 noiseX2dTEB=[None,None,None],
                 noiseY2dTEB=[None,None,None],
                 kBeamX = None,
                 kBeamY = None,
                 doCurl=False,
                 TOnly=False,
                 gradCut=None,
                 uEqualsL=False):

        '''
        All the 2d fourier objects below are pre-fftshifting. They must be of the same dimension.

        templateLiteMap: any object that contains the attributes Nx, Ny, pixScaleX, pixScaleY specifying map dimensions
        theorySpectraForFilters: an orphics.tools.cmb.TheorySpectra object with CMB Cls loaded
        theorySpectraForNorm=None: same as above but if you want to use a different cosmology in the expected value of the 2-pt
        noiseX2dTEB=[None,None,None]: a list of 2d arrays that corresponds to the noise power in T, E, B (same units as Cls above)
        noiseY2dTEB=[None,None,None]: the same as above but if you want to use a different experiment for the Y maps
        doCurl=False: return curl Omega estimates too? If yes, output of getKappa will be (kappa,curl)
        TOnly=False: do only TT? If yes, others will not be initialized and you'll get errors if you try to getKappa(XY) for XY!=TT
        halo=False: use the halo lensing estimators?
        gradCut=None: if using halo lensing estimators, specify an integer up to what L the X map will be retained

        '''


        # initialize norm and filters

        self.doCurl = doCurl





        self.AL = {}
        if doCurl: self.OmAL = {}

        if kBeamX is not None:           
            self.kBeamX = kBeamX
        else:
            self.kBeamX = 1.
            
        if kBeamY is not None:           
            self.kBeamY = kBeamY
        else:
            self.kBeamY = 1.


        self.N = QuadNormSmooth(shape,wcs,gradCut=gradCut,kBeamX=self.kBeamX,kBeamY=self.kBeamY)

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

        
        for cmb in cmbList:
            if uEqualsL:
                uClFilt = theory_filters.lCl(cmb,self.N.modLMap)
            else:
                uClFilt = theory_filters.uCl(cmb,self.N.modLMap)

            if theory_norm is not None:
                if uEqualsL:
                    uClNorm = theory_filters.lCl(cmb,self.N.modLMap)
                else:
                    uClNorm = theory_norm.uCl(cmb,self.N.modLMap)
            else:
                uClNorm = uClFilt
            lClFilt = theory_filters.lCl(cmb,self.N.modLMap)
            self.N.addUnlensedFilter2DPower(cmb,uClFilt)
            self.N.addLensedFilter2DPower(cmb,lClFilt)
            self.N.addUnlensedNorm2DPower(cmb,uClNorm)
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,noiseX2dTEB[i])
            self.N.addNoise2DPowerYY(noise,noiseY2dTEB[i])

        try:
            self.N.addClkk2DPower(theory_filters.gCl("kk",self.N.modLMap))
        except:
            print "Couldn't add Clkk2d power"
            
        self.OmAL = None
        for est in estList:
            self.AL[est] = self.N.getNlkk2d(est)
            if doCurl: self.OmAL[est] = self.N.getCurlNlkk2d(est)



        

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

        import orphics.tools.io as io
        io.quickPlot2d(np.fft.fftshift((WXY.real)),"wxy_"+XY+".png")
        io.quickPlot2d(np.fft.fftshift((WY.real)),"wyy_"+Y+Y+".png")
        io.quickPlot2d(np.fft.fftshift((self.N.Nlkk[XY])),"nxy_"+XY+".png")

        lx = self.N.lxMap
        ly = self.N.lyMap

        if Y in ['E','B']:
            phaseY = self.phaseY
        else:
            phaseY = 1.

        phaseB = (int(Y=='B')*1.j)+(int(Y!='B'))
        
        fmask = self.N.modLMap*0.+1.
        #fmask[self.N.modLMap>6000]=0.

        HighMapStar = ifft(self.kHigh[Y]*WY*phaseY*phaseB*fmask,axes=[-2,-1],normalize=True).conjugate()
        kPx = fft(ifft(self.kGradx[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])*fmask
        kPy = fft(ifft(self.kGrady[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])*fmask  
        rawKappa = ifft(1.j*lx*kPx + 1.j*ly*kPy,axes=[-2,-1],normalize=True).real

        AL = np.nan_to_num(self.AL[XY]*fmask)


        kappaft = -AL*fft(rawKappa,axes=[-2,-1])*fmask
        self.kappa = ifft(kappaft,axes=[-2,-1],normalize=True).real
        try:
            #raise
            assert not(np.any(np.isnan(self.kappa)))
        except:
            import orphics.tools.io as io
            import orphics.tools.stats as stats
            io.quickPlot2d(self.kappa.real,"nankappa.png")
            debug_edges = np.arange(20,20000,100)
            dbinner = stats.bin2D(self.N.modLMap,debug_edges)
            cents, bclkk = dbinner.bin(self.N.clkk2d)
            cents, nlkktt = dbinner.bin(self.N.Nlkk['TT'])
            try:
                cents, nlkkeb = dbinner.bin(self.N.Nlkk['EB'])
            except:
                pass
            pl = io.Plotter(scaleY='log',scaleX='log')
            pl.add(cents,bclkk)
            pl.add(cents,nlkktt,label="TT")
            try:
                pl.add(cents,nlkkeb,label="EB")
            except:
                pass
            pl.legendOn()
            pl.done("clkk.png")

            sys.exit()
        
            
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


        if self.doCurl:
            OmAL = self.OmAL[XY]
            rawCurl = ifft(1.j*lx*kPy - 1.j*ly*kPx,axes=[-2,-1],normalize=True).real
            self.curl = -ifft(OmAL*fft(rawCurl,axes=[-2,-1]),axes=[-2,-1],normalize=True)
            return self.kappa, self.curl



            
        return self.kappa


