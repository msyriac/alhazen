import os
import sys
import numpy as np
import time

i = 0
nrange = np.arange(0.2,10.0,0.2)
for noise in nrange:
    for tellmin,pellmin in [(500,500),(200,50)]:
    
        cmd = "quick_wq.sh python tests/nlDump.py "+str(noise)+" "+ str(tellmin) + " " + str(pellmin) + " & "
        print cmd
        os.system(cmd)
        time.sleep(0.3)
        i+=1
        print i
