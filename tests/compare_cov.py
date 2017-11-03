import numpy as np
import orphics.tools.io as io
import pickle as pickle

pl = io.Plotter(scaleY='log')
TCMB = 2.7255e6
p = []
p.append("/gpfs01/astro/www/msyriac/plots/sims_small_scale_clkk_highres_g_2000_cseed_0_experiment_0.3arc_0.5uk_reconstruction_small_unlensed_False_noise.pkl")
p.append("/gpfs01/astro/www/msyriac/plots/experiment_0.3arc_0.5uk_2000_noise_mpismall.pkl")
for k,pfile in enumerate(p):
    cents,n1d = pickle.load(open(pfile,'rb'))
    if k==1: n1d *= TCMB**2.
    pl.add(cents,n1d,label=str(k))
pl.legendOn()
pl.done("testnoise.png")

# cov1 = "/gpfs01/astro/www/msyriac/ForNam/experiment_0.3arc_0.5uk_2000_4sqdeg_covmat_dl300.npy"
# lbin1 = "/gpfs01/astro/www/msyriac/ForNam/experiment_0.3arc_0.5uk_2000_4sqdeg_lbin_edges_dl300.npy"
# #cov1 = "/gpfs01/astro/www/msyriac/ForNam/experiment_0.3arc_0.5uk_4sqdeg_covmat.npy"
# #lbin1 = "/gpfs01/astro/www/msyriac/ForNam/experiment_0.3arc_0.5uk_4sqdeg_lbin_edges.npy"
# fsky1 = 4./41250.


# cov2 = "/gpfs01/astro/www/msyriac/plots/sims_small_scale_clkk_g_2000_cseed_0_experiment_0.3arc_0.5uk_reconstruction_small_covmat_11.6491285568sqdeg.npy"
# lbin2 = "/gpfs01/astro/www/msyriac/plots/sims_small_scale_clkk_g_2000_cseed_0_experiment_0.3arc_0.5uk_reconstruction_small_lbin_edges_11.6491285568sqdeg.npy"
# fsky2 = 11.6491285568/41250.

cov3 = "/gpfs01/astro/www/msyriac/plots/sims_small_scale_clkk_highres_g_2000_cseed_0_experiment_0.3arc_0.5uk_reconstruction_small_unlensed_False_covmat_2.91260385523sqdeg.npy"
lbin3 = "/gpfs01/astro/www/msyriac/plots/sims_small_scale_clkk_highres_g_2000_cseed_0_experiment_0.3arc_0.5uk_reconstruction_small_unlensed_False_lbin_edges_2.91260385523sqdeg.npy"
fsky3 = 2.91260385523/41250.

cov4 = "/gpfs01/astro/www/msyriac/plots/experiment_0.3arc_0.5uk_2000_2.91260385523sqdeg_covmat_dl300.npy"
lbin4 = "/gpfs01/astro/www/msyriac/plots/experiment_0.3arc_0.5uk_2000_2.91260385523sqdeg_lbin_edges_dl300.npy"
fsky4 = 2.91260385523/41250.

pl = io.Plotter(scaleY='log')

i = 0
#for cov,lbin,fsky in zip([cov1,cov2,cov3,cov4],[lbin1,lbin2,lbin3,lbin4],[fsky1,fsky2,fsky3,fsky4]):
for cov,lbin,fsky in zip([cov3,cov4],[lbin3,lbin4],[fsky3,fsky4]):
    covmat = np.load(cov)
    lbin_edges = np.load(lbin)
    print((lbin_edges[0], lbin_edges[-1],np.diff(lbin_edges)[0]))
    lcents = (lbin_edges[1:]+lbin_edges[:-1])/2.
    diag = np.sqrt(np.diagonal(covmat)*lcents*np.diff(lbin_edges)*fsky)

    pl.add(lcents,diag,label=str(i))
    i+=1

pl.legendOn()
pl.done("test.png")



cov3 = "/gpfs01/astro/www/msyriac/plots/experiment_0.3arc_0.5uk_2000_2.91260385523sqdeg_covmat_dl300.npy"
lbin3 = "/gpfs01/astro/www/msyriac/plots/experiment_0.3arc_0.5uk_2000_2.91260385523sqdeg_lbin_edges_dl300.npy"
fsky3 = 2.91260385523/41250.

cov4 = "/gpfs01/astro/www/msyriac/plots/experiment_0.3arc_0.5uk_2000_2.91260385523sqdeg_autocovmat_dl300.npy"
lbin4 = "/gpfs01/astro/www/msyriac/plots/experiment_0.3arc_0.5uk_2000_2.91260385523sqdeg_lbin_edges_dl300.npy"
fsky4 = 2.91260385523/41250.


import orphics.tools.stats as stats
i = 0
for cov,lbin,fsky in zip([cov3,cov4],[lbin3,lbin4],[fsky3,fsky4]):
    covmat = np.load(cov)
    lbin_edges = np.load(lbin)
    lcents = (lbin_edges[1:]+lbin_edges[:-1])/2.
    lmin = lcents[0]
    lmax = lcents[-1]
    corr = stats.cov2corr(covmat)
    io.quickPlot2d(np.rot90(corr),"corr"+str(i)+".pdf",extent=[lmin,lmax,lmin,lmax],ticksize=10,labsize=10,ftsize=10,lim=[0.,1.0])
    i+=1


