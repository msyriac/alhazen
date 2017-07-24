from orphics.tools.io import Plotter
import sys


from enlib import enmap
import numpy as np

from flipper.fft import fft as fft_gen,ifft as ifft_gen


def fft(m):
    return fft_gen(m,axes=[-2,-1])
def ifft(m):
    return ifft_gen(m,axes=[-2,-1],normalize=True)




def kappa_to_phi(kappa,modlmap,return_fphi=False):
    fphi = enmap.samewcs(kappa_to_fphi(kappa,modlmap),kappa)
    phi =  enmap.samewcs(ifft(fphi).real, kappa) 
    if return_fphi:
        return phi, fphi
    else:
        return phi

def kappa_to_fphi(kappa,modlmap):
    return fkappa_to_fphi(fft(kappa),modlmap)

def fkappa_to_fphi(fkappa,modlmap):
    kmap = np.nan_to_num(2.*fkappa/modlmap/(modlmap+1.))
    kmap[modlmap<2.] = 0.
    return kmap




class alphaMaker(object):

    def __init__(self,imap):


        thetaMap = imap.posmap()
        thetaModSqMap = np.sum(thetaMap**2,0)


        kernels = thetaMap/thetaModSqMap/np.pi

        self.ftkernels = fft_gen(kernels,axes=[-2,-1])




    def kappaToAlpha(self,kappaMap,test=False):
        

        fKappa = fft_gen(kappaMap,axes=[-2,-1])
        fAlpha = self.ftkernels * fKappa
        pixScaleY, pixScaleX = kappaMap.pixshape()
        Ny,Nx = kappaMap.shape

        #retAlpha = (np.fft_gen.ifft_genshift(enmap.ifft_gen(fAlpha,normalize=False).real)+kappaMap*0.)*pixScaleY*pixScaleX/Nx/Ny
        retAlpha = -(np.fft_gen.ifft_genshift(ifft_gen(fAlpha,axes=[-2,-1],normalize=False).real[::-1])+kappaMap*0.)*pixScaleY*pixScaleX/Nx/Ny
        
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


