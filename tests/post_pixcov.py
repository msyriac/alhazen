print("starting imports...")
import numpy as np
from enlib import enmap,resample,lensing
import orphics.analysis.flatMaps as fmaps
from alhazen.halos import NFWkappa
from szar.counts import ClusterCosmology
import orphics.tools.io as io
import alhazen.lensTools as lt
from mpi4py import MPI
import sys,os
import pickle as pickle
import enlib.fft as fftfast
print("finished imports...")

def nfwkappa(massOverh):
    sgn = 1. if massOverh>0. else -1.
    zL = 0.7
    overdensity = 180.
    critical = False
    atClusterZ = False
    concentration = 3.2
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    kappa,r500 = NFWkappa(cc,np.abs(massOverh),concentration,zL,pa.modrmap* 180.*60./np.pi,winAtLens,
                          overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    return sgn*kappa

from alhazen.maxlike import lnlike

out_dir = os.environ['WWW']+"plots/maxlike_hdv_nodim_"

lmax = 8000
cc = ClusterCosmology(lmax=lmax,pickling=True)
theory = cc.theory

arc = 10.0
px = 0.5
lens_order = 5


shape,wcs = enmap.get_enmap_patch(arc,px,proj="car")


noise_T = 1.0
    
pa = fmaps.PatchArray(shape,wcs,dimensionless=False,skip_real=False)
pa.add_theory(cc,theory,lmax)
pa.add_gaussian_beam(1.0)
pa.add_white_noise_with_atm(noise_T,0.,0,1,0,1)

Npoints = 60
mrange = np.linspace(1,3,Npoints)*1.e14


trueM = 2e14
kappa = nfwkappa(trueM)
phi, fphi = lt.kappa_to_phi(kappa,pa.modlmap,return_fphi=True)
#grad_phi = enmap.grad(phi)
alpha_pix = enmap.grad_pixf(fphi)

N = 10000
Nfor = N

totlikes = 0.
allike = 1.

Ms = []
logdets = []
cinvs = []

TCMB = 1. #2.7255e6
from scipy.linalg import pinv2

for k,M in enumerate(mrange):

    M,cov = pickle.load(open(out_dir+"c_"+str(k)+".pkl",'rb'))
    covnoise = np.diag(np.ones(cov.shape[0]))*(noise_T*np.pi/180./60./TCMB)**2.
    cov = cov + covnoise

    
    s,logdet = np.linalg.slogdet(cov)
    print((k,M,s,logdet,np.linalg.cond(cov)))
    assert s>0
        
    Ms.append(  M )
    logdets.append( logdet )
    cinvs.append( pinv2(cov) )

mrange = np.array(Ms)
    
for i in range(N):
    lnlikes = []
    cmb_map = pa.get_unlensed_cmb(seed=2*i+100000000)
    lensed = lensing.lens_map_flat_pix(cmb_map, alpha_pix,order=lens_order) if np.abs(M)>1.e-3 else cmb_map
    flensed = fftfast.fft(lensed,axes=[-2,-1])
    flensed *= pa.lbeam
    lensed = fftfast.ifft(flensed,axes=[-2,-1],normalize=True).real
    noise = pa.get_noise_sim(seed=2*i+1+100000000)
    measured = lensed + noise

    for k,M in enumerate(mrange):
    
        logdet = logdets[k]
        cinv = cinvs[k]
        
        lnlikeval = lnlike(logdet,cinv,measured)


        lnlikes.append(lnlikeval)


    nlnlikes = -0.5*np.array(lnlikes)
    totlikes += nlnlikes.copy()
    if i%100==0:
        print(i)
    
pl = io.Plotter()

Npoints = 500
pmin = 1.e14
pmax = 3.e14
mfinerange = np.linspace(pmin,pmax,Npoints)

totlikes -= totlikes.max()
degmax = 2
p = np.polyfit(mrange,totlikes,deg=degmax)
pfunc = lambda x: sum([c*x**(degmax-k) for k,c in enumerate(p)])

quad = np.abs(p[degmax-2])
sigwidth = np.sqrt(1./quad/2.)*np.sqrt(N/Nfor)


mmax = mrange[np.argmax(totlikes)]
print((mmax,sigwidth))

pl.add(mrange,totlikes,marker="x")
pl.add(mfinerange,pfunc(mfinerange))

pl._ax.axvline(x=trueM,ls="--")
pl._ax.axvline(x=mmax,ls="-")
pl.done(out_dir+"chisq.png")

pl = io.Plotter()

likes = np.exp(totlikes)
likes /= likes.sum()
pl.add(mrange,likes/likes.max(),marker="x",alpha=0.2)

print(("S/N : ",mmax/sigwidth))
print(("Bias % : ",(mmax-trueM)*100./trueM))
print(("Bias sigma : ",(mmax-trueM)/sigwidth))


fitlike = np.exp(-(mfinerange-mmax)**2./2./sigwidth**2.)
pl.add(mfinerange,fitlike/fitlike.max())
pl._ax.axvline(x=trueM,ls="--")
pl._ax.axvline(x=mmax,ls="-")
pl._ax.set_xlim(pmin,pmax)
pl.done(out_dir+"lnlikes.png")
