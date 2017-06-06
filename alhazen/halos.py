import orphics.analysis.flatMaps as fmaps
import numpy as np
from orphics.tools.stats import bin2D,getStats,timeit,getAmplitudeLikelihood
from scipy.optimize import curve_fit as cfit
from scipy.stats import norm
from scipy.interpolate import splrep,splev,interp1d
from scipy.fftpack import fftshift,ifftshift,fftfreq
from scipy.integrate import simps
import orphics.tools.io as io
import flipper.liteMap as lm

import flipper.fft as fftfast

# g(x) = g(theta/thetaS) HuDeDeoVale 2007
gnfw = lambda x: np.piecewise(x, [x>1., x<1., x==1.], \
                            [lambda y: (1./(y*y - 1.)) * \
                             ( 1. - ( (2./np.sqrt(y*y - 1.)) * np.arctan(np.sqrt((y-1.)/(y+1.))) ) ), \
                             lambda y: (1./(y*y - 1.)) * \
                            ( 1. - ( (2./np.sqrt(-(y*y - 1.))) * np.arctanh(np.sqrt(-((y-1.)/(y+1.)))) ) ), \
                        lambda y: (1./3.)])

f_c = lambda c: np.log(1.+c) - (c/(1.+c))


#@timeit
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



#@timeit
def NFWMatchedFilterSN(clusterCosmology,log10Moverh,c,z,ells,Nls,kellmax,overdensity=500.,critical=True,atClusterZ=True,arcStamp=100.,pxStamp=0.05,saveId=None,verbose=False,rayleighSigmaArcmin=None,returnKappa=False,winAtLens=None):
    if rayleighSigmaArcmin is not None: assert rayleighSigmaArcmin>=pxStamp
    M = 10.**log10Moverh

    lmap = lm.makeEmptyCEATemplate(raSizeDeg=arcStamp/60., decSizeDeg=arcStamp/60.,pixScaleXarcmin=pxStamp,pixScaleYarcmin=pxStamp)
    kellmin = 2.*np.pi/arcStamp*np.pi/60./180.
    
    xMap,yMap,modRMap,xx,yy = fmaps.getRealAttributes(lmap)
    lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(lmap)
    
        
    cc = clusterCosmology

    cmb = False
    if winAtLens is None:
        cmb = True
        comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
        comL = cc.results.comoving_radial_distance(z)*cc.h
        winAtLens = (comS-comL)/comS

    kappaReal, r500 = NFWkappa(cc,M,c,z,modRMap*180.*60./np.pi,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
    
    dAz = cc.results.angular_diameter_distance(z) * cc.h
    th500 = r500/dAz
    #fiveth500 = 10.*np.pi/180./60. #5.*th500
    fiveth500 = 5.*th500
    # print "5theta500 " , fiveth500*180.*60./np.pi , " arcminutes"
    # print "maximum theta " , modRMap.max()*180.*60./np.pi, " arcminutes"

    kInt = kappaReal.copy()
    kInt[modRMap>fiveth500] = 0.
    # print "mean kappa inside theta500 " , kInt[modRMap<fiveth500].mean()
    # print "area of th500 disc " , np.pi*fiveth500**2.*(180.*60./np.pi)**2.
    # print "estimated integral " , kInt[modRMap<fiveth500].mean()*np.pi*fiveth500**2.
    k500 = simps(simps(kInt, yy), xx)
    
    if verbose: print "integral of kappa inside disc ",k500
    kappaReal[modRMap>fiveth500] = 0. #### !!!!!!!!! Might not be necessary!
    # if cmb: print z,fiveth500*180.*60./np.pi
    Ukappa = kappaReal/k500


    
    # pl = Plotter()
    # pl.plot2d(Ukappa)
    # pl.done("output/kappa.png")

    ellmax = kellmax
    ellmin = kellmin

    
    
    Uft = fftfast.fft(Ukappa,axes=[-2,-1])

    if rayleighSigmaArcmin is not None:
        Prayleigh = rayleigh(modRMap*180.*60./np.pi,rayleighSigmaArcmin)
        outDir = "/gpfs01/astro/www/msyriac/plots/"
        # io.quickPlot2d(Prayleigh,outDir+"rayleigh.png")
        rayK = fftfast.fft(ifftshift(Prayleigh),axes=[-2,-1])
        rayK /= rayK[modLMap<1.e-3]
        Uft = Uft.copy()*rayK
    
    Upower = np.real(Uft*Uft.conjugate())

    

    # pl = Plotter()
    # pl.plot2d(fftshift(Upower))
    # pl.done("output/upower.png")


    
    Nls[Nls<0.]=0.
    s = splrep(ells,Nls,k=3)
    Nl2d = splev(modLMap,s) 
    
    Nl2d[modLMap<ellmin]=np.inf
    Nl2d[modLMap>ellmax] = np.inf

    area = lmap.Nx*lmap.Ny*lmap.pixScaleX*lmap.pixScaleY
    Upower = Upower *area / (lmap.Nx*lmap.Ny)**2
        
    filter = np.nan_to_num(Upower/Nl2d)
    #filter = np.nan_to_num(1./Nl2d)
    filter[modLMap>ellmax] = 0.
    filter[modLMap<ellmin] = 0.
    # pl = Plotter()
    # pl.plot2d(fftshift(filter))
    # pl.done("output/filter.png")
    # if (cmb): print Upower.sum()
    # if not(cmb) and z>2.5:
    #     bin_edges = np.arange(500,ellmax,100)
    #     binner = bin2D(modLMap, bin_edges)
    #     centers, nl2dells = binner.bin(Nl2d)
    #     centers, upowerells = binner.bin(np.nan_to_num(Upower))
    #     centers, filterells = binner.bin(filter)
    #     from orphics.tools.io import Plotter
    #     pl = Plotter(scaleY='log')
    #     pl.add(centers,upowerells,label="upower")
    #     pl.add(centers,nl2dells,label="noise")
    #     pl.add(centers,filterells,label="filter")
    #     pl.add(ells,Nls,ls="--")
    #     pl.legendOn(loc='upper right')
    #     #pl._ax.set_ylim(0,1e-8)
    #     pl.done("output/filterells.png")
    #     sys.exit()
    
    varinv = filter.sum()
    std = np.sqrt(1./varinv)
    sn = k500/std
    if verbose: print sn

    if saveId is not None:
        np.savetxt("data/"+saveId+"_m"+str(log10Moverh)+"_z"+str(z)+".txt",np.array([log10Moverh,z,1./sn]))

    if returnKappa:
        return sn,fftfast.ifft(Uft,axes=[-2,-1],normalize=True).real*k500
    return sn, k500, std



    

def rayleigh(theta,sigma):
    sigmasq = sigma*sigma
    #return np.exp(-0.5*theta*theta/sigmasq)
    return theta/sigmasq*np.exp(-0.5*theta*theta/sigmasq)
        


def NFWkappa(cc,massOverh,concentration,zL,thetaArc,winAtLens,overdensity=500.,critical=True,atClusterZ=True):

    comL  = (cc.results.comoving_radial_distance(zL) )*cc.h

    

    c = concentration
    M = massOverh

    zdensity = 0.
    if atClusterZ: zdensity = zL

    if critical:
        r500 = cc.rdel_c(M,zdensity,overdensity).flatten()[0] # R500 in Mpc/h
    else:
        r500 = cc.rdel_m(M,zdensity,overdensity) # R500 in Mpc/h


    conv=np.pi/(180.*60.)
    theta = thetaArc*conv # theta in radians

    rS = r500/c

    thetaS = rS/ comL 


    const12 = 9.571e-20 # 2G/c^2 in Mpc / solar mass 
    fc = np.log(1.+c) - (c/(1.+c))    
    #const3 = comL * comLS * (1.+zL) / comS #  Mpc
    const3 = comL *  (1.+zL) *winAtLens #  Mpc
    const4 = M / (rS*rS) #solar mass / MPc^2
    const5 = 1./fc
    

    kappaU = gnfw(theta/thetaS)+theta*0. # added for compatibility with enmap

    consts = const12 * const3 * const4 * const5
    kappa = consts * kappaU


    return kappa, r500




def getDLnMCMB(ells,Nls,clusterCosmology,log10Moverh,z,concentration,arcStamp,pxStamp,arc_upto,bin_width,expectedSN,Nclusters=1000,numSims=30,saveId=None,numPoints=1000,nsigma=8.,overdensity=500.,critical=True,atClusterZ=True):

    import flipper.liteMap as lm
    if saveId is not None: from orphics.tools.output import Plotter

    M = 10.**log10Moverh

    cc = clusterCosmology

    stepfilter_ellmax = max(ells)
    

    lmap = lm.makeEmptyCEATemplate(raSizeDeg=arcStamp/60., decSizeDeg=arcStamp/60.,pixScaleXarcmin=pxStamp,pixScaleYarcmin=pxStamp)

    xMap,yMap,modRMap,xx,xy = fmaps.getRealAttributes(lmap)
    lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(lmap)

    kappaMap,retR500 = NFWkappa(cc,M,concentration,z,modRMap*180.*60./np.pi,winAtLens,overdensity,critical,atClusterZ)
    finetheta = np.arange(0.01,arc_upto,0.01)
    finekappa,retR500 = NFWkappa(cc,M,concentration,z,finetheta,winAtLens,overdensity,critical,atClusterZ)
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

    sn = ampBest/ampErr/np.sqrt(numSims)
    snAll = ampBest/ampErr
    if snAll<5.: print "WARNING: ", saveId, " run with mass ", M , " and redshift ", z , " has overall S/N<5. \
    Consider re-running with a greater numSims, otherwise estimate of per Ncluster S/N will be noisy."

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

        np.savetxt("data/"+saveId+"_m"+str(log10Moverh)+"_z"+str(z)+".txt",np.array([log10Moverh,z,1./sn]))
    
    return 1./sn


# NFW dimensionless form
fnfw = lambda x: 1./(x*((1.+x)**2.))
Gval = 4.517e-48 # Newton G in Mpc,seconds,Msun units
cval = 9.716e-15 # speed of light in Mpc,second units

# NFW density (M/L^3) as a function of distance from center of cluster
def rho_nfw(M,c,R):
    return lambda r: 1./(4.*np.pi)*((c/R)**3.)*M/f_c(c)*fnfw(c*r/R)

# NFW projected along line of sight (M/L^2) as a function of angle on the sky in radians
def proj_rho_nfw(theta,comL,M,c,R):
    thetaS = R/c/comL
    return 1./(4.*np.pi)*((c/R)**2.)*M/f_c(c)*(2.*gnfw(theta/thetaS))

# Generic profile projected along line of sight (M/L^2) as a function of angle on the sky in radians
# rhoFunc is density (M/L^3) as a function of distance from center of cluster
@timeit
def projected_rho(thetas,comL,rhoFunc,pmaxN=2000,numps=500000):
    # default integration times are good to 0.01% for z=0.1 to 3
    # increase numps for lower z/theta and pmaxN for higher z/theta
    # g(x) = \int dl rho(sqrt(l**2+x**2)) = g(theta/thetaS)
    pzrange = np.linspace(-pmaxN,pmaxN,numps)
    g = np.array([np.trapz(rhoFunc(np.sqrt(pzrange**2.+(theta*comL)**2.)),pzrange) for theta in thetas])
    return g


def kappa_nfw(theta,z,comLMpcOverh,M,c,R,windowAtLens):
    return 4.*np.pi*Gval*(1+z)*comLMpcOverh*windowAtLens*proj_rho_nfw(theta,comLMpcOverh,M,c,R)/cval**2.

def kappa_generic(theta,z,comLMpcOverh,rhoFunc,windowAtLens,pmaxN=2000,numps=500000):
    # default integration times are good to 0.01% for z=0.1 to 3
    # increase numps for lower z/theta and pmaxN for higher z/theta
    return 4.*np.pi*Gval*(1+z)*comLMpcOverh*windowAtLens*projected_rho(theta,comLMpcOverh,rhoFunc,pmaxN,numps)/cval**2.


