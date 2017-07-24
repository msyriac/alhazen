import astropy.io.fits as fits
import numpy as np

cat_file = "/astro/astronfs01/workarea/msyriac/act/ACTPol.fits"

h = fits.open(cat_file)
print h[1].header
sys.exit()
print h[1].columns
m500 = h[1].data['M500_M500']
z = h[1].data['redshift']
print m500.size
hasm = m500[m500>1.e-3]
z = z[m500>1.e-3]
print hasm.size
print hasm[z>1.e-3].size
