import time
import os
import numpy as np

Mrange = np.arange(1.,10,0.5)*1e14
for M in Mrange:
    cmd = "quick_mpi.py 8 all python -W ignore tests/pix_cov.py -M "+str(M)
    print cmd
    os.system(cmd)
    time.sleep(0.05)

