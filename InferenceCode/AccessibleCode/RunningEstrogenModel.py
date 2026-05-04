from inspect import Parameter
import sys

 
import numpy as np
import matplotlib.pyplot as plt

import _EstrogenModel 
import pdb 
import pickle
import os
import random 


import pickle
plt.rcParams.update({ "text.usetex": True, "font.family": "Helvetica"})

plt.rcParams.update({  "text.usetex": True, "font.family": "sans-serif", "font.sans-serif": "Helvetica",})
plt.rcParams['font.size'] = '10'; plt.rcParams['axes.linewidth'] = '1'
plt.rcParams['axes.spines.left'] = True;  plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.top'] = False; plt.rcParams['axes.spines.right'] = False
plt.rcParams.update({  "text.usetex": True, "font.family": "STIXGeneral","font.sans-serif": ["Helvetica"], "font.size":12, "savefig.pad_inches":'0.1' })#, "savefig.bbox":"tight" })

 
  
if __name__ == '__main__':


    model = _EstrogenModel.EstrogenModel();   
    print(model.GetInitialConditions())


    t_span = (0, 100)
    t_eval = np.linspace(*t_span, 1000)



    sol = model.simulate([], t_eval)


    # Extract variables
    Ca_0 = sol[0]
    Ca_in = sol[1]
    Ca_SR = sol[2]
    V = sol[3]
    y_g = sol[4]
    F = sol[5]
                   

                    # -----------------------------
    # Plotting
    # -----------------------------
    plt.figure(figsize=(10, 4))

    plt.subplot(1,4,1)

    plt.plot(t_eval, Ca_0, label="Ca_0")
    plt.ylabel(r"$Ca_{0}$")
    plt.xlabel("Time")
    plt.subplot(1,4,2)

    plt.plot(t_eval, Ca_in, label="Ca_in")
    plt.ylabel(r"$Ca_{in}$")
    plt.xlabel("Time")
    plt.subplot(1,4,3)

    plt.plot(t_eval, Ca_SR, label="Ca_SR")
    plt.ylabel(r"$Ca_{SR}$")
    plt.xlabel("Time")
    plt.subplot(1,4,4)

    plt.plot(t_eval, y_g, label="y_g")
    plt.ylabel(r"$y_g$")
    plt.xlabel("Time")
    # plt.ylim(0,2)
    # plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(2,1,1)

    plt.plot(t_eval, F, label="Force (F)")
    plt.title("Contraction Force")
    plt.xlabel("Time")
    # plt.legend()
    plt.subplot(2,1,2)

    plt.plot(t_eval, V, label="Voltage (V)")
    plt.title("Voltage")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

 