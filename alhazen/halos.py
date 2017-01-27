import orphics.analysis.flatMaps as fmaps
import numpy as np
from orphics.tools.stats import bin2D,getStats,timeit,getAmplitudeLikelihood
from scipy.optimize import curve_fit as cfit
from scipy.stats import norm

@timeit
def getProfiles(generator,stepfilter_ellmax,kappaMap,binner,N):
    profiles = []
    totstamp = 0.
    for i in range(N):
        noise = generator.getMap(stepFilterEll=stepfilter_ellmax)
        stamp = kappaMap + noise
        totstamp += stamp
        centers, profile = binner.bin(stamp)
        
        profiles.append(profile)
    return profiles,totstamp/N

    
def predictSN(polComb,noiseTY,noisePY,N,MM):
    pol = polComb[1]
    if pol=='T':
        noisePred=noiseTY
        noiseVal = 1.0
    else:
        noisePred=noisePY
        noiseVal = 1.414

    snList = {}
    snList['TT'] = 15.3
    snList['TE'] = 1.5
    snList['ET'] = 4.8
    snList['EB'] = 15.4
    snList['TB'] = 4.7
    snList['EE'] = 5.1
    return (MM/2.e14)*((noiseVal/noisePred))*snList[polComb]*np.sqrt(N/1000.)



def NFWkappa(cc,massOverh,concentration,zL,thetaArc,overdensity=500.,critical=True,atClusterZ=True): #theta in arcminutes


    gnfw = lambda x: np.piecewise(x, [x>1., x<1., x==1.], \
                                [lambda y: (1./(y*y - 1.)) * \
                                 ( 1. - ( (2./np.sqrt(y*y - 1.)) * np.arctan(np.sqrt((y-1.)/(y+1.))) ) ), \
                                 lambda y: (1./(y*y - 1.)) * \
                                ( 1. - ( (2./np.sqrt(-(y*y - 1.))) * np.arctanh(np.sqrt(-((y-1.)/(y+1.)))) ) ), \
                            lambda y: (1./3.)])




    cmbZ = cc.cmbZ
    comL  = cc.results.comoving_radial_distance(zL) 
    comS  = cc.results.comoving_radial_distance(cmbZ) 
    comLS = comS-comL

    

    c = concentration
    M = massOverh*cc.h

    zdensity = 0.
    if atClusterZ: zdensity = zL

    if critical:
        r500 = cc.rdel_c(M,zdensity,overdensity).flatten()[0] *cc.h # R500 in Mpc No 1/h
    else:
        r500 = cc.rdel_m(M,zdensity,overdensity) *cc.h # R500 in Mpc No 1/h


    conv=np.pi/(180.*60.)
    theta = thetaArc*conv # theta in radians
    rS = r500/c

    thetaS = rS/ comL 


    const12 = 9.571e-20 # 2G/c^2 in Mpc / solar mass 
    fc = np.log(1.+c) - (c/(1.+c))    
    const3 = comL * comLS * (1.+zL) / comS #  Mpc
    const4 = M / (rS*rS) #solar mass / MPc^2
    const5 = 1./fc
    

    kappaU = gnfw(theta/thetaS)
    consts = const12 * const3 * const4 * const5
    kappa = consts * kappaU


    return kappa




def getDLnMCMB(ells,Nls,clusterCosmology,M,z,concentration,arcStamp,pxStamp,arc_upto,bin_width,expectedSN,Nclusters=1000,numSims=30,saveId=None,numPoints=1000,nsigma=8.,overdensity=500.,critical=True,atClusterZ=True):

    import flipper.liteMap as lm
    if saveId is not None: from orphics.tools.output import Plotter

    cc = clusterCosmology

    stepfilter_ellmax = max(ells)
    

    lmap = lm.makeEmptyCEATemplate(raSizeDeg=arcStamp/60., decSizeDeg=arcStamp/60.,pixScaleXarcmin=pxStamp,pixScaleYarcmin=pxStamp)

    xMap,yMap,modRMap,xx,xy = fmaps.getRealAttributes(lmap)
    lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(lmap)

    kappaMap = NFWkappa(cc,M,concentration,z,modRMap*180.*60./np.pi,overdensity,critical,atClusterZ)
    finetheta = np.arange(0.01,arc_upto,0.01)
    finekappa = NFWkappa(cc,M,concentration,z,finetheta,overdensity,critical,atClusterZ)
    kappaMap = fmaps.stepFunctionFilterLiteMap(kappaMap,modLMap,stepfilter_ellmax)

    generator = fmaps.GRFGen(lmap,ells,Nls)
    
    bin_edges = np.arange(0.,arc_upto,bin_width)
    binner = bin2D(modRMap*180.*60./np.pi, bin_edges)
    centers, thprof = binner.bin(kappaMap)


    if saveId is not None:
        pl = Plotter()
        pl.plot2d(kappaMap)
        pl.done("output/"+saveId+"kappa.png")

    
    expectedSNGauss = expectedSN*np.sqrt(numSims)
    sigma = 1./expectedSNGauss
    amplitudeRange = np.linspace(1.-nsigma*sigma,1.+nsigma*sigma,numPoints)

    lnLikes = 0.
    bigStamp = 0.
    for i in range(numSims):
        profiles,totstamp = getProfiles(generator,stepfilter_ellmax,kappaMap,binner,Nclusters)
        bigStamp += totstamp
        stats = getStats(profiles)
        if i==0 and (saveId is not None):
            pl = Plotter()
            pl.add(centers,thprof,lw=2,color='black')
            pl.add(finetheta,finekappa,lw=2,color='black',ls="--")
            pl.addErr(centers,stats['mean'],yerr=stats['errmean'],lw=2)
            pl._ax.set_ylim(-0.01,0.3)
            pl.done("output/"+saveId+"profile.png")

            pl = Plotter()
            pl.plot2d(totstamp)
            pl.done("output/"+saveId+"totstamp.png")


        Likes = getAmplitudeLikelihood(stats['mean'],stats['covmean'],amplitudeRange,thprof)
        lnLikes += np.log(Likes)


    width = amplitudeRange[1]-amplitudeRange[0]

    Likes = np.exp(lnLikes)
    Likes = Likes / (Likes.sum()*width) #normalize
    ampBest,ampErr = cfit(norm.pdf,amplitudeRange,Likes,p0=[1.0,0.5])[0]


    if saveId is not None:
        Fit = np.array([np.exp(-0.5*(x-ampBest)**2./ampErr**2.) for x in amplitudeRange])
        Fit = Fit / (Fit.sum()*width) #normalize
        pl = Plotter()
        pl.add(amplitudeRange,Likes,label="like")
        pl.add(amplitudeRange,Fit,label="fit")
        pl.legendOn(loc = 'lower left')
        pl.done("output/"+saveId+"like.png")
        pl = Plotter()
        pl.plot2d(bigStamp/numSims)
        pl.done("output/"+saveId+"bigstamp.png")

    sn = ampBest/ampErr/np.sqrt(numSims)
    snAll = ampBest/ampErr
    if snAll<5.: print "WARNING: ", saveId, " run with mass ", M , " and redshift ", z , " has overall S/N<5. \
    Consider re-running with a greater numSims, otherwise estimate of per Ncluster S/N will be noisy."
    
    return 1./sn
