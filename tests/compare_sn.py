import numpy as np
import cPickle as pickle
from orphics.theory.cosmology import Cosmology

cc = Cosmology(lmax=6000,pickling=True)
fsky = 2.91260385523/41250.

#fine_ells = np.arange(2,
clkk = cc.theory.gCl('kk',fine_ells)

p = []
p.append("/gpfs01/astro/www/msyriac/plots/sims_small_scale_clkk_highres_g_2000_cseed_0_experiment_0.3arc_0.5uk_reconstruction_small_unlensed_False_REPLACE_2.91260385523sqdeg.npy")
p.append("/gpfs01/astro/www/msyriac/plots/experiment_0.3arc_0.5uk_2000_2.91260385523sqdeg_REPLACE_dl300.npy")
for k,pfile in enumerate(p):

    covfile = pfile.replace("REPLACE","covmat")
    lbinfile = pfile.replace("REPLACE","lbin_edges")

    cov = np.load(covfile)
    lbin_edges = np.load(lbinfile)

    print cov.shape
    print lbin_edges.shape


