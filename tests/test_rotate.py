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
pix = 2.0
shape,wcs = enmap.rect_geometry(width_arcmin=wdeg*60.,px_res_arcmin=pix,height_arcmin=hdeg*60.,yoffset_degree=yoffset)


ells = np.arange(0,5000,1)
ps = theory.lCl('TT',ells).reshape((1,1,5000))

with bench.show("Map generation"):
    mg = enmap.MapGen(shape,wcs,ps)
    taper,w2 = fmaps.get_taper(shape,taper_percent = 8.0,pad_percent = 2.0,weight=None)
    map_south = mg.get_map()*taper

with bench.show("Rotation init"):
    r = fmaps.MapRotatorEquator(map_south.shape,map_south.wcs,wdeg,hdeg,verbose=True,width_multiplier=0.6,height_multiplier=1.2)
with bench.show("Rotation init"):
    rotmap = r.rotate(map_south)

print((map_south.shape))
print((rotmap.shape))
    
io.highResPlot2d(map_south,"smap.png")
io.highResPlot2d(rotmap,"rotmap.png")
