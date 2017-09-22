import sys, os
import numpy as np
import argparse
from enlib import enmap
from orphics.theory.cosmology import Cosmology
import orphics.tools.cmb as cmb
import orphics.analysis.flatMaps as fmaps
import orphics.tools.io as io
import orphics.tools.stats as stats
import numpy as np
import alhazen.quadraticEstimator as qe
from scipy.interpolate import interp1d

# output
out_dir = os.environ['WWW']+"plots/"

# lmax
lmax_global = 8500
lmax = 6000
kellmin = 10
kellmax = lmax
tellmin = 10
tellmax = 6000
pellmin = tellmin
pellmax = tellmax
niter = 12

# cosmology
cc = Cosmology(lmax=lmax_global,pickling=True,dimensionless=False)
theory = cc.theory
ellrange = np.arange(0,lmax_global,1)
clkk = theory.gCl('kk',ellrange)
lcltt = theory.lCl('TT',ellrange)
ucltt = theory.uCl('TT',ellrange)

# quadratic estimator
deg = 5.
px = 0.5
shape, wcs = enmap.get_enmap_patch(deg*60.,px,proj="car",pol=False)
template = fmaps.simple_flipper_template_from_enmap(shape,wcs)
kbeam = np.zeros(shape)+1.
lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape,wcs)
lbin_edges = np.arange(kellmin,kellmax,200)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
fmask = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=kellmin,lmax=kellmax)
nlgen = qe.NlGenerator(template,theory)
nlgen.updateNoise(beamX=0.,noiseTX=0.,noisePX=0.,tellminX=tellmin,tellmaxX=tellmax,pellminX=pellmin,pellmaxX=pellmax)
nlgen.N.getNlkk2d('TT',halo=True)
nlkk0_2d = nlgen.N.Nlkk['TT']
cents,nlkk0 = lbinner_dat.bin(nlkk0_2d)
clkk2d = theory.gCl('kk',modlmap_dat)


# lensed cls prediction
dtheory = cmb.get_lensed_cls(theory,ellrange,clkk,lmax)
dlcltt = dtheory.lCl('TT',ellrange)
ducltt = dtheory.uCl('TT',ellrange)

plkk = io.Plotter(scaleY='log')
plkk.add(ellrange,clkk,color="k",lw=3)
plkk.add(cents,nlkk0,ls="--")#,alpha=1/(niter+3.))

pltt = io.Plotter(scaleY='log')
pltt.add(ellrange,ucltt*ellrange**2.,ls="--")
pltt.add(ellrange,lcltt*ellrange**2.,ls="-")
#pltt.add(ellrange,dlcltt*ellrange**2.,ls="-.")

# unlensed limit
nlgen = qe.NlGenerator(template,theory,lensedEqualsUnlensed=True)
nlgen.updateNoise(beamX=0.,noiseTX=0.,noisePX=0.,tellminX=tellmin,tellmaxX=tellmax,pellminX=pellmin,pellmaxX=pellmax)
nlgen.N.getNlkk2d('TT',halo=True)
nlkku2d = nlgen.N.Nlkk['TT']
cents,nlkku = lbinner_dat.bin(nlkku2d)
plkk.add(cents,nlkku,ls="-",lw=3,color="red")



# iterate this
nlkk2d = nlkk0_2d
clkk_now2d = clkk2d
for i in range(niter):
    print "Iteration ", i
    alpha = (i+2.)/(niter+3.)
    wiener2d = clkk2d/(clkk2d+nlkk2d)
    clkk_now2d = (1.-wiener2d)*clkk_now2d
    cents, clkk_now1d = lbinner_dat.bin(clkk_now2d)
    clkk_now1d = interp1d(cents,clkk_now1d,fill_value="extrapolate")(ellrange)
    plkk.add(ellrange,clkk_now1d,color="k",alpha=alpha,ls="-.")
    dtheory = cmb.get_lensed_cls(theory,ellrange,clkk_now1d,lmax)
    dlcltt = dtheory.lCl('TT',ellrange)
    pltt.add(ellrange,dlcltt*ellrange**2.,ls="-.",alpha=alpha)
    nlgen = qe.NlGenerator(template,dtheory)
    nlgen.updateNoise(beamX=0.,noiseTX=0.,noisePX=0.,
                      tellminX=tellmin,tellmaxX=tellmax,pellminX=pellmin,pellmaxX=pellmax)
    nlgen.N.getNlkk2d('TT',halo=True)
    nlkk2d = nlgen.N.Nlkk['TT']
    cents,nlkk = lbinner_dat.bin(nlkk2d)
    plkk.add(cents,nlkk,ls="--",alpha=alpha)


pltt._ax.set_xlim(2,lmax)
pltt.done(out_dir+"cltt.png")
plkk._ax.set_xlim(2,kellmax)
plkk._ax.set_ylim(1.e-10,1.e-6)
plkk.done(out_dir+"clkk.png")

