import numpy as np


def lnlike(logdet,covinv,stamp):

    
    Npix = stamp.size
    assert covinv.size==Npix**2

    # print "Calculating norm of lnlike..."
    #lognorm = -0.5*(Npix*np.log(2.*np.pi)+np.trace(np.log(cov)))
    #lognorm = np.trace(np.log(cov))
    # s,logdet = np.linalg.slogdet(cov)
    # #print s,logdet
    # try:
    #     assert s>0
    # except:
    #     print s,logdet
    #     sys.exit()
    lognorm = logdet
    # print lognorm
    
    vec = stamp.reshape((Npix,1))

    #quad = -np.dot(np.dot(vec.T,covinv),vec)
    quad = np.dot(np.dot(vec.T,covinv),vec)
    # print quad

    # sys.exit()

    ans = lognorm+quad
    assert ans.size==1
    return ans[0,0]
