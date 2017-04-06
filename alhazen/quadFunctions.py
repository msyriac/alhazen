import numpy as np

def fXY(XY,theory,ll1,ll2,l1,l2,cos2phi=None,sin2phi=None):
    C = theory.uCl

    if XY=='TT':
        return ll1*C('TT',l1)+ll2*C('TT',l2)
    elif XY=='TE':
        return ll1*cos2phi*C('TE',l1)+ll2*C('TE',l2)
    elif XY=='EE':
        return (ll1*C('EE',l1)+ll2*C('EE',l2))*cos2phi
    elif XY=='ET':
        return ll2*cos2phi*C('TE',l2)+ll1*C('TE',l1)
    elif XY=='EB':
        return ll1*C('EE',l1)*sin2phi
    elif XY=='TB':
        return ll1*C('TE',l1)*sin2phi

def F(XY,f,fS,theory,NlfuncdictX,NlfuncdictY,ll1,ll2,l1,l2,cos2phi=None,sin2phi=None,halo=False,gradCut=None):
    X,Y = XY

    if halo:
        if Y=='T':
            cfact = 1.
        elif Y=='E':
            cfact = cos2phi
        elif Y=='B':
            cfact = sin2phi
        else:
            raise ValueError
        YY = Y+Y
        WXYl1 = WXY(XY,theory,NlfuncdictX,l1)
        if gradCut is not None:
            WXYl1[l1>gradCut]=0.
        WYl2 = WY(YY,theory,NlfuncdictY,l2)
        
        return ll1*WXYl1*WYl2*cfact

    
    if XY in ['TT','EE']:
        return 0.5*f*np.nan_to_num(1./(theory.lCl(X+X,l1)+NlfuncdictX[X+X](l1)))*np.nan_to_num(1./(theory.lCl(Y+Y,l2)+NlfuncdictY[Y+Y](l2)))
    elif XY in ['EB','TB']:
        return f*np.nan_to_num(1./(theory.lCl(X+X,l1)+NlfuncdictX[X+X](l1)))*np.nan_to_num(1./(theory.lCl(Y+Y,l2)+NlfuncdictY[Y+Y](l2)))
    elif XY=='TE':

        C_EE = lambda ell: np.nan_to_num(theory.lCl('EE',ell)+NlfuncdictY['EE'](ell))
        C_TT = lambda ell: np.nan_to_num(theory.lCl('TT',ell)+NlfuncdictX['TT'](ell))
        C_TE = lambda ell: theory.lCl('TE',ell)
        
        C_EE_l1 = C_EE(l1)
        C_TT_l2 = C_TT(l2)
        C_TE_l1 = C_TE(l1)
        C_TE_l2 = C_TE(l2)

        prod1 = C_TE_l1*C_TE_l2
        prod2 = C_EE_l1*C_TT_l2

        retval = np.nan_to_num(prod2*f - prod1*fS) *np.nan_to_num(1./ ( (C_TT(l1)*C_EE(l2)*prod2) - (prod1*prod1)))
        return retval
    elif XY=='ET':
        ftemp = fS.copy()
        fS = f.copy()
        f = ftemp.copy()

        C_EE = lambda ell: np.nan_to_num(theory.lCl('EE',ell)+NlfuncdictX['EE'](ell))
        C_TT = lambda ell: np.nan_to_num(theory.lCl('TT',ell)+NlfuncdictY['TT'](ell))
        C_TE = lambda ell: theory.lCl('TE',ell)
        
        C_EE_l2 = C_EE(l2)
        C_TT_l1 = C_TT(l1)
        C_TE_l1 = C_TE(l1)
        C_TE_l2 = C_TE(l2)

        prod1 = C_TE_l2*C_TE_l1
        prod2 = C_EE_l2*C_TT_l1

        return np.nan_to_num(prod2*f - prod1*fS) *np.nan_to_num(1./ ( (C_TT(l2)*C_EE(l1)*prod2) - (prod1*prod1)))

    
def WXY(XY,theory,Nlfuncdict,l1):

    X,Y = XY
    if Y=='B': Y='E'
    gradClXY = X+Y
    if XY=='ET': gradClXY = 'TE'
    W = np.nan_to_num(theory.uCl(gradClXY,l1)/(theory.lCl(X+X,l1)+Nlfuncdict[X+X](l1)))

    return W

def WY(YY,theory,Nlfuncdict,l2):
    assert YY[0]==YY[1]
    W = np.nan_to_num(1./(theory.lCl(YY,l2)+Nlfuncdict[YY](l2)))
    return W


def crossIntegrand(alphaXY,betaXY,theory,NlfuncdictX,NlfuncdictY,Falpha,FBeta,FBetaS,l1,l2,independentExperiments=False):
    Xalpha, Yalpha = alphaXY
    Xbeta, Ybeta = betaXY
    
    def totC(XY,Nlfuncdict,indCheck):
        X,Y = XY
        YY = Y+Y
        if XY=='EB' or XY=='BE' or XY=='TB' or XY=='BT': return lambda ell: 0.
        if X==Y and not(indCheck):
            noise = Nlfuncdict[YY]
        else:
            noise = lambda ell: 0.
            
        return lambda ell: np.nan_to_num(theory.lCl(XY,ell)+noise(ell))
    
    C_x1x2 = totC(Xalpha+Xbeta,NlfuncdictX,indCheck=False)(l1)
    C_y1y2 = totC(Yalpha+Ybeta,NlfuncdictY,indCheck=False)(l2)
    C_x1y2 = totC(Xalpha+Ybeta,NlfuncdictX,indCheck=independentExperiments)(l1) # if they are the same experiment, it shouldn't
    C_y1x2 = totC(Yalpha+Xbeta,NlfuncdictY,indCheck=independentExperiments)(l2) # matter if I pass X or Y noise
    
 
                

    return np.nan_to_num(Falpha*(FBeta*C_x1x2*C_y1y2+FBetaS*C_x1y2*C_y1x2))


