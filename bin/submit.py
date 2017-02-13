import os
import sys
import numpy as np
import time
from shutil import copyfile
from ConfigParser import SafeConfigParser 


timestamp = str(time.time())
iniFile = "output/submission"+timestamp+".ini"
copyfile("input/submission.ini",iniFile) 

Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

numJobs = Config.getint('jobs','numJobs')
numClusters = Config.getint('jobs','numClusters')
snapRange = [int(x) for x in Config.get('jobs','snapRange').split(',')]
saveName = Config.get('jobs','saveName')

for snap in range(snapRange[0],snapRange[1]):
    for i in range(numJobs):

        cmd = "quick_wq.sh python bin/lensRecon.py "+str(i)+" "+str(numClusters)+" "+str(snap)+" "+saveName+" & "
        print cmd
        os.system(cmd)
        time.sleep(0.5)
