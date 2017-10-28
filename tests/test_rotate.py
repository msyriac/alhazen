from enlib import enmap, coordinates, interpol, bench
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
    taper,w2 = fmaps.get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
    map_south = mg.get_map()*taper

    



class MapRotator(object):
    def __init__(self,shape_source,wcs_source,shape_target,wcs_target):
        self.pix_target = get_rotated_pixels(shape_source,wcs_source,shape_target,wcs_target)
    def rotate(self,imap):
        return rotate_map(imap,pix_target=self.pix_target)

class MapRotatorEquator(MapRotator):
    def __init__(self,shape_source,wcs_source,patch_width,patch_height,width_multiplier=1.,
                 height_multiplier=1.5,pix_target_override_arcmin=None,proj="car",verbose=False):
        
        if pix_target_override_arcmin is None:
            input_dec = enmap.posmap(shape_source,wcs_source)[0]
            max_dec = np.max(np.abs(input_dec))
            del input_dec
            recommended_pix = pix*np.cos(max_dec)

            if verbose:
                print "INFO: Maximum declination in southern patch : ",max_dec*180./np.pi, " deg."
                print "INFO: Recommended pixel size for northern patch : ",recommended_pix, " arcmin"

        else:
            recommended_pix = pix_target_override_arcmin
            
        shape_target,wcs_target = enmap.rect_geometry(width_arcmin=width_multiplier*patch_width*60.,
                                                      height_arcmin=height_multiplier*patch_height*60.,
                                                      px_res_arcmin=recommended_pix,yoffset_degree=0.,proj=proj)

        self.target_pix = recommended_pix
        self.source_pix =  np.min(enmap.extent(shape_source,wcs_source)/shape_source[-2:])*60.*180./np.pi
        self.wcs_target = wcs_target
        if verbose:
            print "INFO: Source pixel : ",self.source_pix, " arcmin"
        

        MapRotator.__init__(self,shape_source,wcs_source,shape_target,wcs_target)

    def rotate(self,imap):
        rotated = MapRotator.rotate(self,imap)
        return rotated
    
def get_rotated_pixels(shape_source,wcs_source,shape_target,wcs_target,inverse=False):
    """ Given a source geometry (shape_source,wcs_source)
    return the pixel positions in the target geometry (shape_target,wcs_target)
    if the source geometry were rotated such that its center lies on the center
    of the target geometry.

    WARNING: Only currently tested for a rotation along declination from one CAR
    geometry to another CAR geometry.
    """

    # what are the center coordinates of each geometris
    center_source = enmap.pix2sky(shape_source,wcs_source,(shape_source[0]/2.,shape_source[1]/2.))
    center_target= enmap.pix2sky(shape_target,wcs_target,(shape_target[0]/2.,shape_target[1]/2.))
    decs,ras = center_source
    dect,rat = center_target

    # what are the angle coordinates of each pixel in the target geometry
    pos_target = enmap.posmap(shape_target,wcs_target)
    lra = pos_target[1,:,:].ravel()
    ldec = pos_target[0,:,:].ravel()
    del pos_target

    # recenter the angle coordinates of the target from the target center to the source center
    if inverse:
        newcoord = coordinates.decenter((lra,ldec),(rat,dect,ras,decs))
    else:
        newcoord = coordinates.recenter((lra,ldec),(rat,dect,ras,decs))
    del lra
    del ldec

    # reshape these new coordinates into enmap-friendly form
    new_pos = np.empty((2,shape_target[0],shape_target[1]))
    new_pos[0,:,:] = newcoord[1,:].reshape(shape_target)
    new_pos[1,:,:] = newcoord[0,:].reshape(shape_target)
    del newcoord

    # translate these new coordinates to pixel positions in the target geometry based on the source's wcs
    pix_new = enmap.sky2pix(shape_source,wcs_source,new_pos)

    return pix_new

def rotate_map(imap,shape_target=None,wcs_target=None,pix_target=None):
    if pix_target is None:
        pix_target = get_rotated_pixels(shape_source,wcs_source,shape_target,wcs_target)
    else:
        assert (shape_target is None) and (wcs_target is None), "Both pix_target and shape_target,wcs_target must not be specified."

    rotmap = enmap.at(imap,pix_target,unit="pix")
    return rotmap


with bench.show("Rotation init"):
    r = MapRotatorEquator(map_south.shape,map_south.wcs,wdeg,hdeg,verbose=True,width_multiplier=0.6,height_multiplier=1.2)
with bench.show("Rotation init"):
    rotmap = r.rotate(map_south)

io.highResPlot2d(map_south,"smap.png")
io.highResPlot2d(rotmap,"rotmap.png")
