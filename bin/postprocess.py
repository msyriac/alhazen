print "Importing modules..."
import matplotlib
matplotlib.use('Agg')
from enlib import enmap,utils,lensing,powspec
import numpy as np
from alhazen.halos import NFWkappa
from alhazen.lensTools import alphaMaker
from ConfigParser import SafeConfigParser 
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from szlib.szcounts import ClusterCosmology
from orphics.tools.stats import bin2D
from szlib.sims import BattagliaSims
import sys,os
print "Done importing modules..."


iniFile = "input/submission.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

numClusters = Config.getint('jobs','numClusters')
snapRange = [int(x) for x in Config.get('jobs','snapRange').split(',')]
saveName = Config.get('jobs','saveName')

outDir = os.environ['WWW']+"plots/"
# numClusters = int(sys.argv[1])
# snap = int(sys.argv[2])
# saveName = sys.argv[3]

for snap in range(*snapRange):

    kappaStack = 0.
    inputKappaStack = 0.
    szStack = 0.

    N = numClusters

    for i in range(N):
        print i

        kappa = enmap.read_map(saveName+"_kappa_"+str(i)+"_"+str(snap)+".hdf")
        inputKappaMap = enmap.read_map(saveName+"_inpkappa_"+str(i)+"_"+str(snap)+".hdf")
        szMap = enmap.read_map(saveName+"_sz_"+str(i)+"_"+str(snap)+".hdf")

        kappaStack += kappa
        inputKappaStack += inputKappaMap
        szStack += szMap



    pl = Plotter()
    pl.plot2d(kappaStack/N)
    pl.done(outDir+"recon_"+str(snap)+".png")

    pl = Plotter()
    pl.plot2d(inputKappaStack/N)
    pl.done(outDir+"truestack_"+str(snap)+".png")

    pl = Plotter()
    pl.plot2d(szStack/N)
    pl.done(outDir+"szstack_"+str(snap)+".png")


