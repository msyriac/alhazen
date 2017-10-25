from enlib import enmap, curvedsky, lensing
import orphics.tools.io as io
import orphics.tools.cmb as cmb
import orphics.analysis.flatMaps as fmaps
import numpy as np
import healpy as hp

deg = 20.
px = 2.0

shape,wcs = enmap.rect_geometry(deg*60.,px,proj="car",pol=False)
modlmap = enmap.modlmap(shape,wcs)


lxmap,lymap,modlmap,angmap,lx,ly = fmaps.get_ft_attributes_enmap(shape,wcs)
amajor = 3000
bminor = 100
angle = np.pi/4.
r_square = ((lxmap)*np.cos(angle)+(lymap)*np.sin(angle))**2./amajor**2.+((lxmap)*np.sin(angle)-(lymap)*np.cos(angle))**2./bminor**2.
elfact = (1.+1.e3*np.exp(-r_square))
#elfact = 1.
p2d = elfact * cmb.white_noise_with_atm_func(modlmap,uk_arcmin=10.0,lknee=4000,alpha=-4.5,dimensionless=False,TCMB=2.7255e6) 
p2d[modlmap<90]=0.
p2d[modlmap>4000]=0.
io.quickPlot2d(np.fft.fftshift(np.log10(p2d)),io.dout_dir+"p2d.png")
mg = enmap.MapGen(shape,wcs,p2d.reshape(1,1,modlmap.shape[0],modlmap.shape[1]))
imap = mg.get_map()

io.quickPlot2d(imap,io.dout_dir+"cmb.png")
# imap2 = mg.get_map()
# io.quickPlot2d(imap2,io.dout_dir+"cmb2.png")
dtype = np.complex128
rtype = np.zeros([0],dtype=dtype).real.dtype

print "alms..."
alm = curvedsky.map2alm(imap,lmax=4000)
powfac = 0.5
ps_data = enmap.multi_pow(alm*alm.conj()[None,None],powfac).astype(rtype)
print alm
print alm.shape

pspec = np.ones((8000))
pspec[:200] = 0.
pspec[4000:] = 0.
rand_alm,ainfo = curvedsky.rand_alm(ps=pspec,lmax=4000,return_ainfo=True)
#rand_map = curvedsky.rand_map(shape,wcs,pspec)
#rand_alm = curvedsky.map2alm(rand_map,lmax=4000)
ps_alm = enmap.multi_pow(rand_alm*rand_alm.conj()[None,None],powfac).astype(rtype)

new_sim_alm = rand_alm *np.nan_to_num(ps_data/ps_alm)

new_sim = curvedsky.alm2map(new_sim_alm,imap)#,ainfo=ainfo)
#new_sim = curvedsky.alm2map(alm,imap)#,ainfo=ainfo)
io.quickPlot2d(imap,io.dout_dir+"cmbsim.png")

