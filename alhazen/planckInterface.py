import pyfits

def getCatalogRADecsPlanck(fitsLoc='/astro/astronfs01/workarea/msyriac/PlanckClusters/HFI_PCCS_SZ-MMF3_R2.08.fits',sncut=6.):
    
    hdulist = pyfits.open(fitsLoc)
    hdulist.info()


    cols = hdulist[1].columns
    print(cols)
    tbdata = hdulist[1].data

    ras = tbdata.field('RA')
    decs = tbdata.field('DEC')
    #zs = tbdata.field('REDSHIFT')
    snrs = tbdata.field('SNR')

    # print ras
    # print decs
    # print ras.size
    # #print zs[zs>0.].size
    # print snrs
    # print snrs[snrs>6.].size
    # print snrs[snrs>5.].size
    # print snrs[snrs>4.].size
    # print snrs[snrs<4.].size
    
    
    hdulist.close()

    
    return ras[snrs>sncut],decs[snrs>sncut]

def getCatalogRADecsRedmapper(fitsLoc='/astro/astronfs01/workarea/msyriac/PlanckClusters/redmapper_dr8_public_v6.3_catalog.fits',lambda_cut=19.):
    
    hdulist = pyfits.open(fitsLoc)
    hdulist.info()


    cols = hdulist[1].columns
    print(cols)
    tbdata = hdulist[1].data

    ras = tbdata.field('RA')
    decs = tbdata.field('DEC')
    #zs = tbdata.field('REDSHIFT')
    lambdas = tbdata.field('lambda')

    # print ras
    # print decs
    # print ras.size
    # #print zs[zs>0.].size
    # print snrs
    # print snrs[snrs>6.].size
    # print snrs[snrs>5.].size
    # print snrs[snrs>4.].size
    # print snrs[snrs<4.].size
    
    
    hdulist.close()

    
    return ras[lambdas>lambda_cut],decs[lambdas>lambda_cut]

