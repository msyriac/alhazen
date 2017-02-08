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




    def kappaToAlpha(self,kappaMap,test=False):
        

        fKappa = enmap.fft(kappaMap,normalize=False)
        fAlpha = self.ftkernels * fKappa

        retAlpha = np.fft.ifftshift(enmap.ifft(fAlpha,normalize=False).real)+kappaMap*0.

        if test:
            pixScale = 0.5*np.pi/180./60.
            #newKap = self.div(retAlpha[::-1,:,:])
            newKap = enmap.div(retAlpha[::-1,:,:])*pixScale**2.
            #newKap = enmap.div(retAlpha[:,:,:])
            #newKap = enmap.div(retAlpha[::-1,20:-20,20:-20])
            #kappaMap = kappaMap[20:-20,20:-20]
            pl = Plotter()
            pl.plot2d(kappaMap)
            pl.done("output/oldKap.png")
            pl = Plotter()
            pl.plot2d(newKap)
            pl.done("output/newKap.png")
            ratio = np.nan_to_num(newKap/kappaMap)
            thetaMap = kappaMap.posmap()
            print thetaMap.shape
            thetaModMap = 60.*180.*(np.sum(thetaMap**2,0)**0.5)/np.pi


            print ratio[thetaModMap<10].mean()
            pl = Plotter()
            pl.plot2d(ratio[20:-20,20:-20])
            pl.done("output/testratio.png")


        return retAlpha


