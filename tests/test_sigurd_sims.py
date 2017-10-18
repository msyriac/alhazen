from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from enlib import enmap, curvedsky, lensing
import orphics.tools.io as io
import orphics.tools.cmb as cmb
import orphics.analysis.flatMaps as fmaps
import numpy as np
import healpy as hp

sim_root = "/gpfs01/astro/workarea/msyriac/data/sims/alex/sky/"

def load_fullsky(sim_root,prefix):


    for k in range(3):
        uiqu_now = enmap.read_map(sim_root+prefix+"_"+str(k)+".fits")
        shape = uiqu_now.shape
        if k==0:
            uiqu = enmap.ndmap(np.empty((3,shape[0],shape[1])),uiqu_now.wcs)
        uiqu[k] = uiqu_now

    return uiqu

def map2power(iqu,lensed=False):
    from orphics.tools.stats import bin2D, bin1D
    bin_edges = np.arange(200,4000,40)
        
    
    print ("Map 2 alm...")
    alm = curvedsky.map2alm(iqu,lmax=5000)
    cls = hp.alm2cl(alm)
    fineells = np.arange(0,cls.shape[1],1)

    print ("Binning...")
    lbinner = bin1D(bin_edges)
    def b(cls):
        ells,cl1d = lbinner.binned(fineells,fineells*cls)/lbinner.binned(fineells,fineells)
        return ells,cl1d


    ells,cltt = b(cls[0,:])
    ells,clee = b(cls[1,:])
    ells,clbb = b(cls[2,:])
    ells,clte = b(cls[3,:])
    ells,cleb = b(cls[4,:])
    ells,cltb = b(cls[5,:])

#     if lensed:
#         ells,tcltt = b(theory.lCl('TT',fineells))
#         ells,tclee = b(theory.lCl('EE',fineells))
#         ells,tclte = b(theory.lCl('TE',fineells))
#         ells,tclbb = b(theory.lCl('BB',fineells))
#     else:        
#         ells,tcltt = b(theory.uCl('TT',fineells))
#         ells,tclee = b(theory.uCl('EE',fineells))
#         ells,tclte = b(theory.uCl('TE',fineells))
#         ells,tclbb = b(fineells*0.)

#     pl = io.Plotter(scaleY='log')
#     pl.add(ells,cltt*ells**2.)
#     pl.add(ells,clee*ells**2.)
#     pl.add(ells,tcltt*ells**2.,color="k")
#     pl.add(ells,tclee*ells**2.,color="k")
#     if lensed:
#         pl.add(ells,clbb*ells**2.)        
#         pl.add(ells,tclbb*ells**2.,color="k")
#     pl._ax.set_ylim(1.e-3,5e5)
#     pl.done(io.dout_dir+"sigurd_alex_clcomp_lensed_"+str(lensed)+".png")


#     pl = io.Plotter(labelX="$\\ell$",labelY="$\Delta C_{\\ell}/C_{\\ell}$")
#     if True:#(sshape is not None) and (swcs is not None):
#         pl.add(mells,mcltt,label="TT cut-sky",ls="--")
#         pl.add(mells,mclee,label="EE cut-sky",ls="--")
#     pl.add(ells,np.nan_to_num((cltt-tcltt)/tcltt),label="TT Alex")
#     #pl.add(ells,np.nan_to_num((clte-tclte)/tclee),label="TE")
#     pl.add(ells,np.nan_to_num((clee-tclee)/tclee),label="EE Alex")
#     if lensed: pl.add(ells,np.nan_to_num((clbb-tclbb)/tclbb),label="BB Alex")
#     pl._ax.set_xlim(2,4000)
#     pl._ax.set_ylim(-0.1,0.1)
#     pl.hline()
#     pl.legendOn(loc="lower left")
#     pl.done(io.dout_dir+"sigurd_alex_clcomp_lensed_"+str(lensed)+"_diff.png")


#     pl = io.Plotter()
#     pl.add(ells,tclte*ells**2.,color="k")
#     pl.add(ells,clte*ells**2.)
#     pl.add(ells,cleb*ells**2.)
#     pl.add(ells,cltb*ells**2.)
    
#     if not(lensed): pl.add(ells,clbb*ells**2.)
#     pl.done(io.dout_dir+"sigurd_alex_clcomp_cross_lensed_"+str(lensed)+".png")

# cambRoot = sim_root + "cosmo2017"
# theory = cmb.loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)
    
# # uiqu = load_alex(sim_root,"uTquMap")
# # print (uiqu.shape)
# # pix_arc = np.sqrt(uiqu.pixsize())*180.*60./np.pi
# # print (pix_arc, " arcmin pixel")

# # map2power(uiqu)

# liqu = load_alex(sim_root,"lTquMap")
# map2power(liqu,lensed=True)

