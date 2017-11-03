import time
import os,sys
import numpy as np

Npoints = 60
for k in range(Npoints):
    cmd = "quick_mpi.py 4 all python -W ignore tests/pix_cov.py -M "+str(k)
    print(cmd)
    os.system(cmd)
    time.sleep(0.05)
