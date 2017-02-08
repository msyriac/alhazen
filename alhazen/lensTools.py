from orphics.tools.io import Plotter
import sys


from enlib import enmap
import numpy as np

from pyfftw.interfaces.scipy_fftpack import fft2
from pyfftw.interfaces.scipy_fftpack import ifft2



class alphaMaker(object):

    def __init__(self,imap):


        '''
        This is an internal function that will store the lensing kernel for the given stamp dimensions
        when the simulator is initialized.
        '''

        thetaMap = imap.posmap()
        thetaModSqMap = np.sum(thetaMap**2,0)


        kernels = thetaMap/thetaModSqMap/np.pi

        self.ftkernels = enmap.fft(kernels,normalize=False)




    def kappaToAlpha(self,kappaMap,test=False,px=0.5):
        

        fKappa = enmap.fft(kappaMap,normalize=False)
        fAlpha = self.ftkernels * fKappa
        print fAlpha.shape
        pixScale = px*np.pi/180./60.
        Nx,Ny = kappaMap.shape

        retAlpha = (np.fft.ifftshift(enmap.ifft(fAlpha,normalize=False).real)+kappaMap*0.)*pixScale*pixScale/Nx/Ny
        
        if test:
            newKap = np.nan_to_num(0.5*enmap.div(retAlpha[::-1,:,:])) #/pixScale
            thetaMap = kappaMap.posmap()
            thetaModMap = 60.*180.*(np.sum(thetaMap**2,0)**0.5)/np.pi
            print "newkappaint ", np.nanmean(newKap[thetaModMap<10.])
            
            pl = Plotter()
            pl.plot2d(kappaMap)
            pl.done("output/oldKap.png")
            pl = Plotter()
            pl.plot2d(newKap)
            pl.done("output/newKap.png")
            ratio = np.nan_to_num(newKap/kappaMap)
            print thetaMap.shape


            print ratio[thetaModMap<5].mean()
            pl = Plotter()
            pl.plot2d(ratio[200:-200,200:-200])
            pl.done("output/testratio.png")


        return retAlpha


