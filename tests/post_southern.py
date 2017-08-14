import cPickle as pickle
import orphics.tools.stats as stats
import orphics.tools.io as io
import os,sys,glob
from orphics.theory.cosmology import Cosmology
import numpy as np

cc = Cosmology(lmax=3000,pickling=True)
ells = np.arange(2,3000,1)
clkktheory = cc.theory.gCl("kk",ells)


output_dir = "/gpfs01/astro/workarea/msyriac/data/sigurd_sims/"


clkkn0pers = []
clkkpers = []
sdppers = []
clttpers = []

pl = io.Plotter(scaleY="log")
pl.add(ells,clkktheory,color="k",lw=3)

for k in range(50):
    print k

    try:
        exp_name = "south"
        cents_pwr,sn0subbed = pickle.load(open(output_dir+exp_name+"_clkk_n0subbed_"+str(k).zfill(2)+".pkl",'rb'))
        cents_pwr,saclkk = pickle.load(open(output_dir+exp_name+"_rawclkk_"+str(k).zfill(2)+".pkl",'rb'))
        cents_pwr,ssdp = pickle.load(open(output_dir+exp_name+"_superdumbn0_"+str(k).zfill(2)+".pkl",'rb'))
        cents,sdcltt = pickle.load(open(output_dir+exp_name+"_cltt_"+str(k).zfill(2)+".pkl",'rb'))

        pl.add(cents_pwr,sn0subbed,alpha=0.4)


        exp_name = "equator"
        cents_pwr,en0subbed = pickle.load(open(output_dir+exp_name+"_clkk_n0subbed_"+str(k).zfill(2)+".pkl",'rb'))
        cents_pwr,eaclkk = pickle.load(open(output_dir+exp_name+"_rawclkk_"+str(k).zfill(2)+".pkl",'rb'))
        cents_pwr,esdp = pickle.load(open(output_dir+exp_name+"_superdumbn0_"+str(k).zfill(2)+".pkl",'rb'))
        cents,edcltt = pickle.load(open(output_dir+exp_name+"_cltt_"+str(k).zfill(2)+".pkl",'rb'))

        pl.add(cents_pwr,en0subbed,alpha=0.4,ls="--")
    except:
        print "skipping"
        continue
    

    clkkn0pers.append((sn0subbed-en0subbed)*100./en0subbed)
    clkkpers.append((saclkk-eaclkk)*100./eaclkk)
    sdppers.append((ssdp-esdp)*100./esdp)
    clttpers.append((sdcltt-edcltt)*100./edcltt)

pl._ax.set_xlim(0,3000)
pl._ax.set_ylim(1.e-9,1.e-6)
pl.done("clkk.png")

clkkn0stats = stats.getStats(clkkn0pers)
clkkstats = stats.getStats(clkkpers)
sdpstats = stats.getStats(sdppers)
clttstats = stats.getStats(clttpers)


pl = io.Plotter(labelX="$L$",labelY="% diff")
pl.addErr(cents_pwr,clkkn0stats['mean'],yerr=clkkn0stats['errmean'],label="sdn0subbed clkk",ls="-")
pl.addErr(cents_pwr,clkkstats['mean'],yerr=clkkstats['errmean'],label="raw clkk",alpha=0.2,ls="-")
pl.addErr(cents_pwr,sdpstats['mean'],yerr=sdpstats['errmean'],label="sdn0",alpha=0.2,ls="-")
pl.addErr(cents,clttstats['mean'],yerr=clttstats['errmean'],label="cltt",ls="-")
pl.legendOn(labsize=10)
pl._ax.set_xlim(0,3000)
pl._ax.set_ylim(-5,20)
pl._ax.axhline(y=0.,ls="--",color="k",alpha=0.5)
pl.done("cldiff.png")




    
