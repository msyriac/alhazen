from enlib import enmap, bench
import orphics.tools.cmb as cmb
import orphics.tools.io as io
import numpy as np
import sys
import orphics.analysis.flatMaps as fmaps

theory_file_root = "data/Aug6_highAcc_CDM"
theory = cmb.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

wdeg = 45.
hdeg = 15.
yoffset = 60.
pix = 4.0
lmax = 2500
shape,wcs = enmap.rect_geometry(width_arcmin=wdeg*60.,px_res_arcmin=pix,height_arcmin=hdeg*60.,yoffset_degree=yoffset)


ells = np.arange(0,7000,1)
ps = theory.lCl('TT',ells).reshape((1,1,7000))

with bench.show("Map generation"):
    mg = enmap.MapGen(shape,wcs,ps)
    taper,w2 = fmaps.get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
    map_south = mg.get_map()*taper
    
with bench.show("Rotation init"):
    r = fmaps.MapRotatorEquator(map_south.shape,map_south.wcs,wdeg,hdeg,verbose=True,width_multiplier=0.6,height_multiplier=1.2,downsample=False)
with bench.show("Rotation"):
    rotmap = r.rotate(map_south)
    del map_south
    rottap = r.rotate(taper)
    del taper


with bench.show("Map generation"):
    tmg = enmap.MapGen(r.shape_final,r.wcs_final,ps)
    map_test = tmg.get_map()*rottap

prefix = "Oct28_"
# io.highResPlot2d(map_south,prefix+"smap.png")
# io.highResPlot2d(rotmap,prefix+"rotmap.png")
# io.highResPlot2d(rottap,prefix+"taper.png")
# io.highResPlot2d(map_test,prefix+"testmap.png")


w2 = np.mean(rottap**2.)

from orphics.tools.stats import bin2D
fc = enmap.FourierCalc(r.shape_final,r.wcs_final)
modlmap = enmap.modlmap(r.shape_final,r.wcs_final)
p2d_rot,_,_ = fc.power2d(rotmap)/w2
p2d,_,_ = fc.power2d(map_test)/w2
p2d_theory = theory.lCl('TT',modlmap)

bin_edges = np.arange(100,lmax,40)
binner = bin2D(modlmap,bin_edges)

cents,crot = binner.bin(p2d_rot)
cents,ctest = binner.bin(p2d)
cents,ctheory = binner.bin(p2d_theory)


pl = io.Plotter()
pl.add(cents,(crot-ctest)/ctest,label="rot vs test")
pl.add(cents,(crot-ctheory)/ctheory,label="rot vs theory")
pl.add(cents,(ctest-ctheory)/ctheory,label="test vs theory")
pl.hline()
pl.legendOn()
pl.done(prefix+"cldiff.png")

