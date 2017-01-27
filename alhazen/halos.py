import numpy as np

def NFWkappa(cc,m500,c500,zL,thetaArc,cmbZ=1100.): #theta in arcminutes


    gnfw = lambda x: np.piecewise(x, [x>1., x<1., x==1.], \
                                [lambda y: (1./(y*y - 1.)) * \
                                 ( 1. - ( (2./np.sqrt(y*y - 1.)) * np.arctan(np.sqrt((y-1.)/(y+1.))) ) ), \
                                 lambda y: (1./(y*y - 1.)) * \
                                ( 1. - ( (2./np.sqrt(-(y*y - 1.))) * np.arctanh(np.sqrt(-((y-1.)/(y+1.)))) ) ), \
                            lambda y: (1./3.)])





    comL  = cc.results.comoving_radial_distance(zL)

    c500  = c500


    comS  = cc.results.comoving_radial_distance(cmbZ)

    comLS = comS-comL


    M = m500
    omegaM = cc.om

    H0 =cc.h * 3.241E-18 #s^-1
    G=4.52E-48 #solar^-1 mpc^3 s^-2
    rhoC0 = 3.*(H0**2.)/(8.*np.pi*G)   #solar / mpc^3

    r500=(3.*M/(4.*np.pi*500.*omegaM*rhoC0))**(1./3.)


    conv=np.pi/(180.*60.)
    theta = thetaArc*conv # theta in radians
    rS = r500/c500

    thetaS = rS/ comL 

    pref=2.*np.pi*1.91E-19 # this is 8piG/c^2 in units of Mpc/solar mass
    fc = np.log(1.+c500) - (c500/(1.+c500))
    const1 = (3./4./np.pi) #dimensionless
    const2 = pref/3. #H0^2 / rhoC0 = 8piG/3/c^2 = pref / 3 in Mpc/solar mass
    const3 = comL * comLS * (1.+zL) / comS #Mpc ############ change back
    const4 = M / (rS*rS) #solar mass / MPc^2
    const5 = 1./fc


    kappaU = gnfw(theta/thetaS)
    consts = const1 * const2 * const3 * const4 * const5
    kappa = consts * kappaU


    return kappa

