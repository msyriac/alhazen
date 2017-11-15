from orphics.tools.mpi import MPI
import alhazen.utils as utils
import argparse
from enlib import enmap


# Parse command line
parser = argparse.ArgumentParser(description='Run south rotation test.')
parser.add_argument("-x", "--patch_width", type=float, default=40., help="Patch width in degrees.")
parser.add_argument("-y", "--patch_height", type=float, default=15., help="Patch height in degrees.")
parser.add_argument("-o", "--yoffset", type=float, default=60., help="Offset in declination of southern patch center.")
parser.add_argument("-p", "--full_sky_pixel", type=float, default=0.5,help="Full sky pixel resolution in arcminutes.")
parser.add_argument("-i", "--pix_inter", type=float, default=None,help="Intermediate patch pixelization.")
parser.add_argument("-l", "--lmax", type=int, default=7000,help="Lmax for full-sky lensing.")
parser.add_argument("-N", "--Nsims", type=int, default=10,help="Number of sims.")
parser.add_argument("-m", "--meanfield", type=str, default=None,help="Meanfield file root.")
args = parser.parse_args()



pipe = utils.RotTestPipeline(full_sky_pix=args.full_sky_pixel,wdeg=args.patch_width,
                             hdeg=args.patch_height,yoffset=args.yoffset,
                             mpi_comm=MPI.COMM_WORLD,nsims=args.Nsims,lmax=args.lmax,pix_intermediate=args.pix_inter)

cmb = {}
ikappa = {}
mlist = ['e','s','r']
mf = {}

for m in mlist:
    if args.meanfield is not None:
        mf[m] = enmap.read_map(args.meanfield+"/meanfield_"+m+".hdf")
    else:
        mf[m] = 0.

for k,index in enumerate(pipe.tasks):

    cmb['s'],cmb['e'],ikappa['s'],ikappa['e'] = pipe.make_sim(index)
    cmb['r'] = pipe.rotator.rotate(cmb['s'])
    ikappa['r'] = pipe.rotator.rotate(ikappa['s'])

    for m in mlist:
        if pipe.rank==0: pipe.logger.info( "Reconstructing...")

        recon = pipe.reconstruct(m,cmb[m]) - mf[m]

        if pipe.rank==0: pipe.logger.info( "Powers...")

        cxc,kcmb,kcmb = pipe.fc[m].power2d(cmb[m])
        rxr,krecon,krecon = pipe.fc[m].power2d(recon)
        rxr /= pipe.w4[m]
        rxi,kinput = pipe.fc[m].f1power(ikappa[m],krecon)
        rxi /= pipe.w3[m]
        ixi = pipe.fc[m].f2power(kinput,kinput)
        ixi /= pipe.w2[m]
        n0 = pipe.qest[m].N.super_dumb_N0_TTTT(cxc)/pipe.w2[m]**2.
        rxr_n0 = rxr - n0

        pipe.mpibox.add_to_stack("meanfield-"+m,recon)
        
        pipe.mpibox.add_to_stats("cmb-"+m,pipe.binner[m].bin(cxc/pipe.w2[m])[1])
        pipe.mpibox.add_to_stats("rxr-"+m,pipe.binner[m].bin(rxr)[1])
        pipe.mpibox.add_to_stats("rxi-"+m,pipe.binner[m].bin(rxi)[1])
        pipe.mpibox.add_to_stats("ixi-"+m,pipe.binner[m].bin(ixi)[1])
        pipe.mpibox.add_to_stats("n0-"+m,pipe.binner[m].bin(n0)[1])
        pipe.mpibox.add_to_stats("rxr-n0-"+m,pipe.binner[m].bin(rxr_n0)[1])

    


        if k==0 and pipe.rank==0:
            import orphics.tools.io as io
            io.highResPlot2d(cmb[m],io.dout_dir+"cmb_"+m+".png")
            io.highResPlot2d(recon,io.dout_dir+"recon_"+m+".png")
    
    
    
if pipe.rank==0: pipe.logger.info( "MPI Collecting...")
pipe.mpibox.get_stacks(verbose=False)
pipe.mpibox.get_stats(verbose=False)

if pipe.rank==0:
    pipe.dump(save_meanfield=(args.meanfield is None))
