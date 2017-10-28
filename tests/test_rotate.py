from enlib import enmap, coordinates, interpol
import orphics.tools.cmb as cmb
import orphics.tools.io as io
import numpy as np
import sys

theory_file_root = "data/Aug6_highAcc_CDM"
theory = cmb.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

wdeg = 60.
hdeg = 40.
yoffset = 60.
pix = 2.0
shape1,wcs1 = enmap.rect_geometry(width_arcmin=1.5*wdeg*60.,height_arcmin=1.5*hdeg*60.,px_res_arcmin=pix,yoffset_degree=0.)
shape2,wcs2 = enmap.rect_geometry(width_arcmin=wdeg*60.,px_res_arcmin=pix,height_arcmin=hdeg*60.,yoffset_degree=yoffset)


ells = np.arange(0,5000,1)
ps = theory.lCl('TT',ells).reshape((1,1,5000))

mg = enmap.MapGen(shape2,wcs2,ps)
map_south = mg.get_map()
map_north = enmap.empty(shape1,wcs1)

center_south = map_south.pix2sky((map_south.shape[0]/2.,map_south.shape[1]/2.))
center_north = map_north.pix2sky((map_north.shape[0]/2.,map_north.shape[1]/2.))

decs,ras = center_south
decn,ran = center_north

print center_south * 180./np.pi
print center_north * 180./np.pi

pos_north = enmap.posmap(shape1,wcs1)
pos_south = enmap.posmap(shape2,wcs2)

pix_north = enmap.pixmap(shape1,wcs1)
pix_south = enmap.pixmap(shape2,wcs2)

print pix_north[0].min(),pix_north[0].max(),pix_north[1].min(),pix_north[1].max(),map_north.shape
print pix_south[0].min(),pix_south[0].max(),pix_south[1].min(),pix_south[1].max(),map_south.shape


lra = pos_north[1,:,:].ravel()
ldec = pos_north[0,:,:].ravel()

newcoord = coordinates.recenter((lra,ldec),(ran,decn,ras,decs))
new_pos = pos_north.copy()
new_pos[0,:,:] = newcoord[1,:].reshape(map_north.shape)
new_pos[1,:,:] = newcoord[0,:].reshape(map_north.shape)
pix_new = map_south.sky2pix(new_pos)



rotmap = enmap.at(map_south,pix_new,unit="pix")

io.quickPlot2d(map_south,"smap.png")
io.quickPlot2d(rotmap,"rotmap.png")
