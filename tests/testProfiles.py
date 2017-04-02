import numpy as np
from szlib.szcounts import ClusterCosmology
from ConfigParser import SafeConfigParser 
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
import alhazen.halos as halos
from scipy.interpolate import interp1d

def npspace(minim,maxim,num,scale="lin"):
    if scale=="lin" or scale=="linear":
        return np.linspace(minim,maxim,num)
    elif scale=="log":
        return np.logspace(np.log10(minim),np.log10(maxim),num)

iniFile = "../SZ_filter/input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

cosmologyName = 'params' # from ini file
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')

cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = "../SZ_filter/data/cltt_lensed_Feb18.txt")


M500 = 5.e14
c500 = 1.18
z = 0.5
delta = 500
R500 = cc.rdel_c(M500,z,delta)
# R500_alt = cc.rdel_c_alt(M500,c500,delta)
# print R500, R500_alt
# sys.exit()
# R500 = R500_alt

Rrange = npspace(0.1,10.*R500,100,"log")

rhofunc = halos.rho_nfw(M500,c500,R500)
rhos = rhofunc(Rrange)

pl = Plotter(scaleY='log',scaleX='log',labelX="$R$ (Mpc/h)",labelY="$\\rho (h^2 M_{\\odot}/{\\mathrm{Mpc}^3})$")
pl.add(Rrange,rhos)
pl.done("output/rhos.png")

comL = cc.results.comoving_radial_distance(z)*cc.h
thetaS = R500/c500/comL
arcmin = 0.1
arcmax = 30.
thetas = npspace(arcmin*np.pi/180./60.,arcmax*np.pi/180./60.,100,'log')
gs = np.array([halos.projected_rho(theta,comL,rhofunc) for theta in thetas])


gsalt = halos.proj_rho_nfw(thetas,comL,M500,c500,R500)

pl = Plotter(scaleY='log',scaleX='log',labelX="$\\theta$ (arcmin)",labelY="$g(\\theta/\\theta_S)$")
pl.add(thetas*180.*60./np.pi,gs)
pl.add(thetas*180.*60./np.pi,gsalt)
pl.done("output/projrhos.png")


zstar = 1100.
comS = cc.results.comoving_radial_distance(zstar)*cc.h


winAtLens = (comS-comL)/comS

kappas = halos.kappa_nfw(thetas,z,comL,M500,c500,R500,winAtLens)
kappas2 = np.array([halos.kappa_generic(theta,z,comL,rhofunc,winAtLens) for theta in thetas])

kappas3,r500Ret = halos.NFWkappa(cc,M500,c500,z,thetas*180.*60./np.pi,zstar,overdensity=delta,critical=True,atClusterZ=True)

pl = Plotter(scaleY='log',scaleX='log',labelX="$\\theta$ (arcmin)",labelY="$\\kappa$")
pl.add(thetas*180.*60./np.pi,kappas)
pl.add(thetas*180.*60./np.pi,kappas2)
pl.add(thetas*180.*60./np.pi,kappas3)
pl.done("output/kappa.png")


pl = Plotter()
pl.add(thetas*180.*60./np.pi,(kappas-kappas2)*100./kappas,label="numerical integration")
pl.add(thetas*180.*60./np.pi,(kappas-kappas3)*100./kappas,label="analytical")
pl.legendOn()
pl.done("output/kappadiff.png")
