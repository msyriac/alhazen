import sys
from enlib import enmap
import numpy as np
import orphics.analysis.flatMaps as fmaps
import orphics.tools.io as oio
import warnings
import logging
logger = logging.getLogger()


def theory_from_config(Config,theory_section,dimensionless=True):
    sec_type = Config.get(theory_section,"cosmo_type")
    lmax = Config.getint(theory_section,"lmax")
    cc = None
    
    if sec_type=="pycamb_params":
        raise NotImplementedError
    elif sec_type=="cluster":
        from szar.counts import ClusterCosmology
        with oio.nostdout():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logger.disabled = True
                cc = ClusterCosmology(lmax=lmax,pickling=True)
                theory = cc.theory
                logger.disabled = False
    elif sec_type=="default":
        from orphics.theory.cosmology import Cosmology
        with oio.nostdout():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logger.disabled = True
                cc = Cosmology(lmax=lmax,pickling=True)
                theory = cc.theory
                logger.disabled = False
        
    elif sec_type=="camb_file":
        cc = None
        import orphics.tools.cmb as cmb
        file_root = Config.get(theory_section,"camb_file_root")
        theory = cmb.loadTheorySpectraFromCAMB(file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=lmax,get_dimensionless=False)
        try:
            cforce = Config.getboolean(theory_section,"cluster_force")
        except:
            cforce = False
        if cforce:
            from szar.counts import ClusterCosmology
            with oio.nostdout():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    logger.disabled = True
                    cc = ClusterCosmology(skipCls=True)
                    logger.disabled = False
            
            
    elif sec_type=="enlib_file":
        import orphics.tools.cmb as cmb
        file_root = Config.get(theory_section,"enlib_file_root")
        theory = cmb.load_theory_spectra_from_enlib(file_root,lpad=lmax)
        cc = None

    else:
        print sec_type
        raise ValueError


    return theory, cc, lmax


def plot_powers(unlensed_map,lensed_map,modlmap,theory,lbinner_dat,out_dir):
    utt2d = fmaps.get_simple_power_enmap(unlensed_map)
    ltt2d = fmaps.get_simple_power_enmap(lensed_map)

    
    iutt2d = theory.uCl("TT",modlmap)
    iltt2d = theory.lCl("TT",modlmap)

    lb = lbinner_dat
    cents,utt = lb.bin(utt2d)
    cents,ltt = lb.bin(ltt2d)
    cents,iutt = lb.bin(iutt2d)
    cents,iltt = lb.bin(iltt2d)

    pl = oio.Plotter()
    pdiff = (utt-iutt)*100./iutt
    pl.add(cents,pdiff)
    pl.done(out_dir+"uclpdiff.png")

    pl = oio.Plotter()
    pdiff = (ltt-iltt)*100./iltt
    pl.add(cents,pdiff)
    pl.done(out_dir+"lclpdiff.png")

def enmaps_from_config(Config,sim_section,analysis_section,pol=False):
    """
    Algorithm for deciding sim and analysis shapes and wcs:

    Check if user has specified a *data* template
        * If yes, use its shape and wcs for the data
        - Determine ratio simpixel/anpixel
        - Upsample by that ratio to make sim template
        * If no, 

    """
    
    pixel_sim = Config.getfloat(sim_section,"pixel_arcmin")
    buffer_sim = Config.getfloat(sim_section,"buffer")
    projection = Config.get(analysis_section,"projection")
    try:
        pt_file = Config.get(analysis_section,"patch_template")
        imap = enmap.read_map(pt_file)
        shape_dat = imap.shape
        wcs_dat = imap.wcs

        res = np.min(imap.extent()/imap.shape[-2:])*60.*180./np.pi
        if np.isclose(pixel_sim,res,1.e-2):
            shape_sim = shape_dat
            wcs_sim = wcs_dat
        else:
            bbox = enmap.box(shape_dat,wcs_dat)
            shape_sim, wcs_sim = enmap.geometry(bbox,res=pixel_sim*np.pi/180./60.,proj=projection)
            
    except:
        pixel_analysis = Config.getfloat(analysis_section,"pixel_arcmin")
        try:
            width_analysis_deg = Config.getfloat(analysis_section,"patch_degrees_width")
        except:
            width_analysis_deg = Config.getfloat(analysis_section,"patch_arcmin_width")/60.
        try:   
            height_analysis_deg = Config.getfloat(analysis_section,"patch_degrees_height")
        except:
            height_analysis_deg = Config.getfloat(analysis_section,"patch_arcmin_height")/60.
        ra_offset = Config.getfloat(analysis_section,"ra_offset")
        dec_offset = Config.getfloat(analysis_section,"dec_offset")


        
        shape_dat, wcs_dat = enmap.get_enmap_patch(width_analysis_deg*60.,pixel_analysis,proj=projection,pol=pol,height_arcmin=height_analysis_deg*60.,xoffset_degree=ra_offset,yoffset_degree=dec_offset)

        if np.abs(buffer_sim-1.)<1.e-3:
            shape_sim, wcs_sim = enmap.get_enmap_patch(width_analysis_deg*60.,pixel_sim,proj=projection,pol=pol,height_arcmin=height_analysis_deg*60.,xoffset_degree=ra_offset,yoffset_degree=dec_offset)
        else:
            raise NotImplementedError, "Buffer !=1 not implemented"

    return shape_sim, wcs_sim, shape_dat, wcs_dat            


def enmap_from_config_section(Config,section,pol=False):
    analysis_section = section
    
    projection = Config.get(analysis_section,"projection")
    try:
        pt_file = Config.get(analysis_section,"patch_template")
        imap = enmap.read_map(pt_file)
        shape_dat = imap.shape
        wcs_dat = imap.wcs

        res = np.min(imap.extent()/imap.shape[-2:])*60.*180./np.pi
            
    except:
        pixel_analysis = Config.getfloat(analysis_section,"pixel_arcmin")
        try:
            width_analysis_deg = Config.getfloat(analysis_section,"patch_degrees_width")
        except:
            width_analysis_deg = Config.getfloat(analysis_section,"patch_arcmin_width")/60.
        try:   
            height_analysis_deg = Config.getfloat(analysis_section,"patch_degrees_height")
        except:
            height_analysis_deg = Config.getfloat(analysis_section,"patch_arcmin_height")/60.
        ra_offset = Config.getfloat(analysis_section,"ra_offset")
        dec_offset = Config.getfloat(analysis_section,"dec_offset")


        
        shape_dat, wcs_dat = enmap.get_enmap_patch(width_analysis_deg*60.,pixel_analysis,proj=projection,pol=pol,height_arcmin=height_analysis_deg*60.,xoffset_degree=ra_offset,yoffset_degree=dec_offset)

    return shape_dat, wcs_dat            


def patch_array_from_config(Config,exp_name,shape,wcs,dimensionless=True,TCMB=2.7255e6,skip_real=False):
    pa = fmaps.PatchArray(shape,wcs,dimensionless=dimensionless,TCMB=TCMB,skip_real=skip_real)
    try:
        bfile = Config.get(exp_name,"beam_file")
        ells,bls = np.loadtxt(bfile,delimiter=",",unpack=True,use_cols=[0,1])
        pa.add_1d_beam(ells,bls,fill_value="extrapolate")
    except:
        fwhm = Config.getfloat(exp_name,"beam")
        pa.add_gaussian_beam(fwhm)

    try:
        n2d_file_T = Config.get(exp_name,"noise_2d_file_T")
        n2d_file_P = Config.get(exp_name,"noise_2d_file_P")
        imapT = enmap.read_map(n2d_file_T)
        imapP = enmap.read_map(n2d_file_P)
        pa.add_noise_2d(nT=imapT,nP=imapP)
    except:
        noise_T = Config.getfloat(exp_name,"noise_T")
        noise_P = Config.getfloat(exp_name,"noise_P")
        lknee_T = Config.getfloat(exp_name,"lknee_T")
        lknee_P = Config.getfloat(exp_name,"lknee_P")
        alpha_T = Config.getfloat(exp_name,"alpha_T")
        alpha_P = Config.getfloat(exp_name,"alpha_P")


        pa.add_white_noise_with_atm(noise_T,noise_P,lknee_T,alpha_T,lknee_P,alpha_P)


    return pa


def get_patch_degrees(Config,section):

    try:
        patch_arcmins = Config.getfloat(section,"patch_arcmins")
        arcmin = True
    except:
        arcmin = False

    try:
        patch_degrees = Config.getfloat(section,"patch_degrees")
        degree = True
    except:
        degree = False


    if arcmin and degree:
        raise ValueError
    elif arcmin:
        return patch_arcmins/60.
    elif degree:
        return patch_degrees
    else:
        print "ERROR: Patch width not specified."
        sys.exit()

    

def ellbounds_from_config(Config,recon_section,min_ell):
    ret = {}
    for rval in ["tellminX","tellmaxX","pellminX","pellmaxX",
                 "tellminY","tellmaxY","pellminY","pellmaxY",
                 "kellmin","kellmax"]:
        conf_val = oio.get_none_or_int(Config,recon_section,rval)
        ret[rval] = min_ell if conf_val is None else conf_val

    return ret


def kappa_grf_generator(theory):

    clkk = theory.gCl("kk",fine_ells)
    clkk.resize((1,1,clkk.size))
    kappa_map = enmap.rand_map(shape_sim[-2:],wcs_sim,cov=clkk,scalar=True)



def kappa_from_config(Config,kappa_section):

    ktype = Config.get(kappa_section,"type")

    if ktype=="cluster_nfw":
        raise NotImplementedError
    elif ktype=="cluster_battaglia":
        raise NotImplementedError
    elif ktype=="grf":
        vary = Config.getboolean(kappa_section,"vary")
        if vary:
            raise NotImplementedError
        else:
            pass
    else:
        raise ValueError
