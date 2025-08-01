#mapWB module preprocesses the weights and biases

import math
import numpy as np
import csv
import random

"""
This script builds upon and extends work presented in the following publication:

Md Hasibul Amin, Mohammed E. Elbtity, and Ramtin Zand. 2023.
IMAC-Sim: A Circuit-level Simulator For In-Memory Analog Computing Architectures.
In Proceedings of the Great Lakes Symposium on VLSI 2023 (GLSVLSI '23),
Association for Computing Machinery, New York, NY, USA, 659–664.
https://doi.org/10.1145/3583781.3590264

We acknowledge that most of the neural network structure, SPICE simulation framework,
and core evaluation code in this file are derived from or adapted based on the
IMAC-Sim framework presented in the above publication. All credit for the foundational
SPICE code and in-memory analog computing simulation logic belongs to the original authors.

We however, adapt the code to fit our own architectural paradigm and to get relevant statistics
from the SPICE simulation. As such, this file as been provided to recreate the Crossbar SPICE
simulations, if such tools are available on whatever machine is being run.
"""

def mapWB(layernum,rlow,rhigh,nodes,data_dir,weight_var):
    for i in range(layernum-1):
        j=i+1
        weight=np.genfromtxt(data_dir+'/'+'W'+str(j)+'.csv',delimiter=',')
        w_flat=np.reshape(weight,nodes[i]*nodes[i+1])
        weight_w=open(data_dir+'/'+'W'+str(j)+'.txt', "w")
        for k in range(len(w_flat)):
            x=float(w_flat[k])
            if (str(x)!='nan'):
                weight_w.write("%f\n"%(float(w_flat[k])))
        weight_w.close()
        
        bias=np.genfromtxt(data_dir+'/'+'B'+str(j)+'.csv',delimiter=',')
        b_flat=np.reshape(bias,nodes[i+1])
        bias_w=open(data_dir+'/'+'B'+str(j)+'.txt', "w")
        for k in range(len(b_flat)):
            x=float(b_flat[k])
            if (str(x)!='nan'):
                bias_w.write("%f\n"%(float(b_flat[k])))
        bias_w.close()
        
        f=open(data_dir+'/'+'W'+str(j)+'.txt',"r")
        wp=open(data_dir+'/'+'posweight'+str(j)+'.txt', "w")
        wn=open(data_dir+'/'+'negweight'+str(j)+'.txt', "w")
        for l in f:
            if (float(l)==1):
                wp.write("%f\n"%(rlow+random.uniform(-1*weight_var*rlow/100,weight_var*rlow/100)))
                wn.write("%f\n"%(rhigh+random.uniform(-1*weight_var*rhigh/100,weight_var*rhigh/100)))
            if (float(l)==-1):
                wp.write("%f\n"%(rhigh+random.uniform(-1*weight_var*rhigh/100,weight_var*rhigh/100)))
                wn.write("%f\n"%(rlow+random.uniform(-1*weight_var*rlow/100,weight_var*rlow/100)))
            if (float(l)==0):
                wp.write("%f\n"%(rlow+random.uniform(-1*weight_var*rlow/100,weight_var*rlow/100)))
                wn.write("%f\n"%(rlow+random.uniform(-1*weight_var*rlow/100,weight_var*rlow/100)))
        wp.close()
        wn.close()
        f.close()
        
        g=open(data_dir+'/'+'B'+str(j)+'.txt',"r")
        bp=open(data_dir+'/'+'posbias'+str(j)+'.txt', "w")
        bn=open(data_dir+'/'+'negbias'+str(j)+'.txt', "w")
        for l in g:
            if (float(l)==1):
                bp.write("%f\n"%(rlow+random.uniform(-1*weight_var*rlow/100,weight_var*rlow/100)))
                bn.write("%f\n"%(rhigh+random.uniform(-1*weight_var*rhigh/100,weight_var*rhigh/100)))
            if (float(l)==-1):
                bp.write("%f\n"%(rhigh+random.uniform(-1*weight_var*rhigh/100,weight_var*rhigh/100)))
                bn.write("%f\n"%(rlow+random.uniform(-1*weight_var*rlow/100,weight_var*rlow/100)))
            if (float(l)==0):
                bp.write("%f\n"%(rlow+random.uniform(-1*weight_var*rlow/100,weight_var*rlow/100)))
                bn.write("%f\n"%(rlow+random.uniform(-1*weight_var*rlow/100,weight_var*rlow/100)))
        bp.close()
        bn.close()
        g.close()
