from orphics.tools.mpi import MPI
import alhazen.utils as utils
import argparse


# Parse command line
parser = argparse.ArgumentParser(description='Run south rotation test.')
parser.add_argument("-x", "--patch_width", type=float, default=40., help="Patch width in degrees.")
parser.add_argument("-y", "--patch_height", type=float, default=15., help="Patch height in degrees.")
parser.add_argument("-o", "--yoffset", type=float, default=60., help="Offset in declination of southern patch center.")
parser.add_argument("-p", "--full_sky_pixel", type=float, default=0.5,help="Full sky pixel resolution in arcminutes.")
parser.add_argument("-i", "--pix_inter", type=float, default=None,help="Intermediate patch pixelization.")
parser.add_argument("-l", "--lmax", type=int, default=7000,help="Lmax for full-sky lensing.")
parser.add_argument("-N", "--Nsims", type=int, default=10,help="Number of sims.")
args = parser.parse_args()



pipe = utils.RotTestPipeline(full_sky_pix=args.full_sky_pixel,wdeg=args.patch_width,
                             hdeg=args.patch_height,yoffset=args.yoffset,
                             mpi_comm=MPI.COMM_WORLD,nsims=args.Nsims,lmax=args.lmax,pix_intermediate=args.pix_inter)

cmb = {}
recon = {}
mlist = ['e','s','r']


for k,index in enumerate(pipe.tasks):

    cmb['s'],cmb['e'] = pipe.make_sim(index)
    cmb['r'] = pipe.rotator.rotate(cmb['s'])

    for m in mlist:
        recon[m] = pipe.reconstruct(m,cmb[m])


    if k==0 and pipe.rank==0:
        import orphics.tools.io as io
        for m in mlist:
            io.highResPlot2d(cmb[m],io.dout_dir+"cmb_"+m+".png")
            io.highResPlot2d(recon[m],io.dout_dir+"recon_"+m+".png")
    
    
    
