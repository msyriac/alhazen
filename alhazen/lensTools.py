from orphics.tools.io import Plotter
import sys


from enlib import enmap
import numpy as np

from flipper.fft import fft,ifft

class alphaMaker(object):

    def __init__(self,imap):


        thetaMap = imap.posmap()
        thetaModSqMap = np.sum(thetaMap**2,0)


        kernels = thetaMap/thetaModSqMap/np.pi

        self.ftkernels = fft(kernels,axes=[-2,-1])




    def kappaToAlpha(self,kappaMap,test=False):
        

        fKappa = fft(kappaMap,axes=[-2,-1])
        fAlpha = self.ftkernels * fKappa
        pixScaleY, pixScaleX = kappaMap.pixshape()
        Ny,Nx = kappaMap.shape

        #retAlpha = (np.fft.ifftshift(enmap.ifft(fAlpha,normalize=False).real)+kappaMap*0.)*pixScaleY*pixScaleX/Nx/Ny
        retAlpha = -(np.fft.ifftshift(ifft(fAlpha,axes=[-2,-1],normalize=False).real[::-1])+kappaMap*0.)*pixScaleY*pixScaleX/Nx/Ny
        
        if test:
            newKap = -np.nan_to_num(0.5*enmap.div(retAlpha)) 
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


