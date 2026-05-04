from inspect import Parameter
import sys

sys.path.append('~/Documents/projects/Ocular_PKPD_Studies/pints/Estrogen_modelling/') 
sys.path.insert(1, '/Users/jess/Documents/projects/Ocular_PKPD_Studies/pints/Estrogen_modelling/')
# import Data


sys.path.insert(1, '/Users/jess/Documents/projects/Ocular_PKPD_Studies/pints/') 
import pints
import pints.toy as toy
import pints.plot
import numpy as np
import matplotlib.pyplot as plt

 
import AdaptedPriors
import pdb
from nanPints import NaNGaussianLogLikelihood
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

 
BaseResultsFolder ='/Users/jess/Documents/projects/Ocular_PKPD_Studies/pints/Estrogen_modelling/InferenceResults/'

#truncated normal distribution function
def truncated_normal(loc=0, scale=1.0, size=100, bounds=[-3,3], rng=None):
    #bounds are in units of sigma, e.g. bounds=[-3,3] means 3 sigma cutoff
    out=np.zeros(size) #initialize output array
    id=np.arange(size)  #initialize ids of all samples
    nrej=size  
    iterct=0
    while nrej > 0:    #repeat until no output out of bounds
        out[id]=rng.normal(loc, scale,size=nrej) #draw from distrib
        id=np.nonzero(np.logical_or(out >  bounds[1] , out <  bounds[0] ))  #ids where out of bound
        #nrej=np.count_nonzero(out < bounds[0])  #count how many are out of bounds
        nrej=len(id[0])
        if iterct == 0:     
            print("initial accepted fraction: ",(size-nrej)/size)
            pass
        #print("iterct", iterct,"nrej: ",nrej)
        iterct+=1
    return out



     
    
def BayesianSetUp(model, ExperimentalTime, ydata, MeanParameters, SigmaParameters, Lower, Upper):
   
    problem = pints.MultiOutputProblem(model, ExperimentalTime, ydata)   
    log_likelihood = NaNGaussianLogLikelihood(problem)
    log_prior = AdaptedPriors.OrderedTrapOnlyTruncatedMultivariateGaussianLogPrior(MeanParameters,  np.diag(SigmaParameters) , Lower, Upper, model)

    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    return log_posterior


# def RunBayesianInferenceSerumData( RunTime = 1000,  MeanParameters= np.array([0.01753068,     2.73665868e-01, 3.33223189e+00,    0.0015,    1.7373 , 0.018, 0.28478003, 0.03582301]),
#                                     SigmaParameters  =  np.array([  3,     0.3,            3,        0.1,          5,     0.009,       0.3,             0.3]) , 
#                                     Lower = np.array([  0,    0.000000001,     1,      0.00000005,   0.001,    0.000001,    0.001,            0.001]),
#                                     Upper = np.array([ 0.1,    1,             10,       0.01,         5,     0.05  ,    8,                 8]) , 
#                                     Folder = "/",   Parallel = True, ChainN = 30):
# #  
#     # [ExperimentalTime,VEGF,  Aflib ,Trap ] = Data.LockheartDataTogether(); 
#     ydata = np.array([ Aflib, Trap]).transpose()
   
#     model = pints.toy.EstrogenModel();  

#     # model.SetSimulationTime(Data.Lockheart_ExperimmentalTime())
    
#     BI=SetupBI.SetupBayesianInference()
#     log_posterior = BI.BayesianSetUp(model, ExperimentalTime, ydata, MeanParameters, SigmaParameters, Lower, Upper)

#     ChainN = np.max([ChainN, int(len(MeanParameters)*1.6)]) 

#     xs = BI.CreateChains(ChainN, MeanParameters,SigmaParameters, Lower, Upper) #, lowerMultiplier=0.999, upperMultiplier=1.001)
#     # pdb.set_trace()
    
#     mcmc = pints.MCMCController(log_posterior,  chains=ChainN, x0=xs,method=pints.DifferentialEvolutionMCMC)#  method=pints.HaarioBardenetACMC) # method=pints.PopulationMCMC)#  method=pints.DifferentialEvolutionMCMC) # SliceDoublingMCMC   # mcmc = pints.MCMCController(log_pdf, 3, xs, method=pints.RelativisticMCMC, sigma0=sigma)  # method=pints.HamiltonianMCMC)
#     mcmc.set_max_iterations(RunTime);  
#     mcmc.set_parallel(Parallel); 
#     mcmc.set_log_interval(iters=200, warm_up=100) 
#     mcmc.PickleResults(PickleResultsBool =True, IterationTrigger=1000, ResultsFolder = Folder)
     

#     print(' Running...')
#     chains = mcmc.run()
#     print('  -------------------  \n ------------------- \n  -------------------  \n -------------------')
#     print('Done!')

#     picklefile = open(Folder+"InferenceResults.pickle", "wb")
#     pickle.dump(chains, picklefile); picklefile.close()

#     print('And saved :) Thank goodness')

#     results = pints.MCMCSummary(chains=chains[:,int(RunTime/2):,:],parameter_names = ParameterListing )  # results = pints.MCMCSummary(chains=chains, time=mcmc.time() ,parameter_names = ParameterListing ) 
 
#     MeanParameters=results.quantiles()[2]

#     picklefile = open(Folder+"InferenceResultsMeanParameters.pickle", "wb")
#     pickle.dump(MeanParameters, picklefile); picklefile.close()
    
   
   

# def PlottingInferenceResults(Folder, MeanParameters, SigmaParameters, Lower, Upper, ParameterListing = [   "$CL_A$", "$CL_T F$", "$CL_A^G$",      "$p_a$",   "$p_t$","$kela$",  "$\sigma_2$",  "$\sigma_3$"], PlotPairwisePosterior = True, PlotTrace = True, PlotModelFit = True, FromArchieve=False, FinalnChains =0):
    
#     BI=SetupBI.SetupBayesianInference();    plt.rcParams['font.size'] = '14';  # chains = np.delete(chains, 34, axis=0) 


#     chains = BI.OpenPickleResults(FromArchieve, Folder); 
#     MeanParametersNew = BI.GetMean(False, Folder, chains, FinalnChains, ParameterListing)
#     print(MeanParametersNew)
#     pdb.set_trace()
#     # MeanParametersNew = [3.22572628e-02, 3.10524777e-02, 3.42294244e+00, 4.84666003e-01, 4.36047789e-01, 6.83697061e-01, 1.23471407e+00, 4.98539905e+00, 5.04395652e+02, 7.00063635e-02,     2.37111954e-01]
 
#     if PlotModelFit ==True:
#         # C_cA, C_cT
#         # MeanParametersNew[0]*=0.1
#         # MeanParametersNew[1]*=0.00000000000000000000000000001

#         # # TissueClearanceScaler
#         # MeanParametersNew[2]*=0.0000000000000000000000000000001

#         # # pa_ct, pt_ct
#         # MeanParametersNew[3]*=0.0001
#         # MeanParametersNew[4]*=0.000001
#         # print(MeanParametersNew)

#         # # TissueToCircPermRatio
#         # print("MeanParametersNew[5] before:", MeanParametersNew[6])
#         # # MeanParametersNew[5]=1
#         # MeanParametersNew[8] = 0

#         # C_cA, C_cT,TissueClearanceScaler,  pa_ct, pt_ct, TissueToCircPermRatio, TissueToCircPermRatio_VEGF,  koff ,kon, Vin = parameters
        
#         BI.PlottingModelFitWithData(Folder, MeanParametersNew )
     
#         print("Have plotted model fit")


#     if PlotTrace ==True:
 
#         [fig, axes]=pints.plot.traceTwoComp(chains[:,FinalnChains:,:],parameter_names = ParameterListing ,ref_parameters =[MeanParameters,SigmaParameters,Lower,Upper,MeanParametersNew],PlotPrior=1)
#         plt.savefig(Folder+"TracePlot_1.jpg",dpi=350)
#         print("Have plotted trace")
 
    
    
    
    
#     if PlotPairwisePosterior ==True: 
#         FinalnChains =-1000; ChainN = chains.shape[0]; RunTime = chains.shape[1];
#         AllChains = chains[0,-FinalnChains:,:]
#         for I in range(ChainN-1): 
#             AllChains = np.concatenate((AllChains,chains[I+1,FinalnChains:, :]), axis=0 )
        
#         [fig, axes]=pints.plot.pairwise(AllChains, kde=True,parameter_names = ParameterListing, ref_parameters =MeanParametersNew ) 
        
#         fig.set_size_inches(9.15, 8.2)
#         plt.subplots_adjust(right =0.976,bottom=0.108,left=0.12, top=0.971, wspace =0.102, hspace=0)

#         plt.savefig(Folder+"PairwisePosteriors_WithVEGF.jpg",dpi=350)
#         print("Have plotted pairwise posterior")
 


def SetParametersAndPrior():
 
    kON= 1.51728518e+00; koFF = 2.96013962e+00
    ParameterListing = [ "$C_cA$", "$C_cT$", "$C_tT$",   "$pa_ct$", "$pt_ct$",    "$pv$" ,  "$k_{off}$",       "$\sigma_A$" ,   "$\sigma_T$"] 

    #  Monkey mean and std Kon and koff
    Kon =  3.36412781 ; Koff = 3.50805636
    Stdkon = 2.16942874; stdkoff = 1.54825579


    
    c_cA = 0.01;  c_cT = 0.01; alpha = 1.5; p_act = 0.4;  p_tct = 0.5; pv = 1; koff =  4; kon =  2.59581979 # # 10#5.76617508e+00
    c_cA = 0.05;
    VinTissue = 5.05940213e+02 

    Koff = 2.42990608e+00; Kon = 4.40303015e+00
    ParameterListing = [ "$C_cA$", "$C_cT$", "alpha",   "$pa_ct$", "$pt_ct$",    "beta" , "beta_VEGF" ,  "$k_{off}$", "$k_{on}$",  "vin",     "$\sigma_A$" ,   "$\sigma_T$"] 
    
 
    ParameterListing = [ "$C_cA$", "$C_cT$", "alpha",   "$pa_ct$", "$pt_ct$",    "beta" , "beta_VEGF" ,  "$k_{off}$", "$k_{on}$",  "vin",     "$\sigma_A$" ,   "$\sigma_T$"] 
 
    Vin_old, C_cv,   C_tv,  pv_tc , sig =  np.array([528, 5.8, 26.6, 4, 0.3]) 
  
    
        # C_cA,  pa_ct,  koff ,kon, Vin = parameters

    C_cA = 1.86115524e-01; pa_ct =  3.52259421e-04; pt_ct = 3.52259421e-04/2;
    C_cT =  C_cA *0.5;
    TissueClearanceScaler = 0.5
    TissueToCircPermRatio = 1
    TissueToCircPermRatio_VEGF = 1
    C_tA= C_cA * TissueClearanceScaler
    C_tT= C_cT * TissueClearanceScaler
    Vin = 600
    C_cA = 0.25
    C_cT = 0.001
    # pa_ct = 0.00035
    Vin = 5.27832080e+02; Cv = 5.84746724e+00
    # From IC 
    # ParameterListing = [ "$V_{in}$", "$C_cv$",   "$C_tv$",  "$pv_tc$",   "$\sigma_V$" ] 
# 
    # Means  [7.51762092e+02 4.84106323e+00 3.88675387e+01 4.12234946e+00 3.61875083e-01]
    # Lower quatile  [7.44808429e+02  3.64816615e+01 2.81140969e+00 1.94160837e-01]
    Vin = 7.51762092e+02
    Cv = 1.90123227e+00


    MeanParameters = np.array([Vin, Cv, C_cA, C_cT,TissueClearanceScaler,  pa_ct, pt_ct,     0.1,   0.1])
    
                            #   Vin, C_cv, C_cA, C_cT,TissueClearanceScaler,  pa_ct, pt_ct,   koff ,kon = parameters
        
    SigmaParameters = np.array([100, 10, 0.5,  0.5, 0.1, 0.01,   0.01,      0.02,   0.05])
 
    Lower= np.array([0,0,0,          0,      0,        0,  0,          0,  0]) 
    Upper =np.array([900,20, 2,           1.5,      2,       0.01     ,  0.008,              0.7, 0.8]) 
    # ParameterListing=["$C_cA$", "$C_cT$", "alpha",   "$pa_ct$", "$pt_ct$",    "beta" , "beta_VEGF" ,  "$k_{off}$", "$k_{on}$",  "vin",     "$\sigma_A$" ,   "$\sigma_T$"] 
    # 
    # Lower= np.array([0,0,0,0,           300,0,0]) 
    # Upper =np.array([20, 10,   10, 10,  8000 ,   2, 1]) 
    
    assert(len(Upper) == len(Lower))
    assert(len(MeanParameters) == len(SigmaParameters))
    assert(len(MeanParameters) == len(Lower))
    # pdb.set_trace()
 
#  ParameterListing = [ "$C_cA$", "$C_cT$", "$\alpha$",   "$pa_ct$", "$pt_ct$",    "$\beta$" ,  "$k_{off}$",       "$\sigma_A$" ,   "$\sigma_T$"] 
 

    return [MeanParameters, SigmaParameters, Lower, Upper]

 

 

if __name__ == '__main__':


    model = pints.toy.EstrogenModel();  
# model.SetSimulationTime(np.linspace(0, 500000, 100))
    print(model.GetInitialConditions())


     
    # RunTime =5000000 
    # ParameterListing = [ "$C_cA$", "$C_cT$", "$C_tT$",   "$pa_ct$", "$pt_ct$",    "$pv$" ,  "$k_{off}$",       "$\sigma_A$" ,   "$\sigma_T$"] 
    # ParameterListing = [ "Vin", "$C_V$","$C_cA$", "$C_cT$", "alpha",   "$pa_ct$", "$pt_ct$",   "$\sigma_A$" ,   "$\sigma_T$"] 
    # # MeanParameters = np.array([Vin, Cv, C_cA, C_cT,TissueClearanceScaler,  pa_ct, pt_ct,   koff ,kon,   0.1,   0.1])
    
    # # ParameterListing = [ "$C_cA$",    "$pa_ct$",    "$k_{off}$", "$k_{on}$",  "vin",     "$\sigma_A$" ,   "$\sigma_T$"] 
 
    # # ParameterListing = [ "$C_cT$",    "$C_tT$",    "kon",  "koff",    "$\sigma_A$",  "$\sigma_T$"]       
    # [MeanParameters, SigmaParameters, Lower, Upper] = SetParametersAndPrior()
    # # pdb.set_trace()
    # References = BaseResultsFolder+'2026/01_05/2/'
    # if os.path.isdir(References) == 0: os.mkdir(References)

    # RunBayesianInferenceSerumData(RunTime, MeanParameters, SigmaParameters, Lower, Upper,  References, 
    #                               Parallel = True, ChainN =40)
   
   
    # PlottingInferenceResults(References, MeanParameters, SigmaParameters, Lower, Upper, ParameterListing ,
    #                          FromArchieve=True, 
    #                          PlotPairwisePosterior = True , PlotTrace = True, PlotModelFit = True, 
    #                          FinalnChains =   00)



    
 