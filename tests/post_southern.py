import cPickle as pickle
import orphics.tools.stats as stats
import orphics.tools.io as io
import os,sys,glob






for i in range(48):
    cents_pwr,n0subbed = pickle.load(open(output_dir+"clkk_n0subbed_"+str(k).zfill(2)+".pkl",'rb'))
    cents_pwr,aclkk = pickle.load(open(output_dir+"rawclkk_"+str(k).zfill(2)+".pkl",'rb'))
    cents_pwr,sdp = pickle.load(open(output_dir+"superdumbn0_"+str(k).zfill(2)+".pkl",'rb'))
    cents,dcltt = pickle.load(open(output_dir+"cltt_"+str(k).zfill(2)+".pkl",'rb'))
    


