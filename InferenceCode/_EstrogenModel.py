import numpy as np
import pints
from scipy.integrate import solve_ivp 
from . import ToyModel 
import pdb 
import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d, CubicSpline
from numba import njit

import sys
sys.path.append('/Users/jess/Documents/projects/Ocular_PKPD_Studies/pints/BabyAntiVEGF/Growth/')
# import Data
# import GrowthCurves






import warnings
warnings.filterwarnings("error", category=RuntimeWarning)



class EstrogenModel(pints.ForwardModel, ToyModel):
    """
    The whole body PK/PD model for aflibercept in preterm babies
    """

    def __init__(self, y0=None):
        super(EstrogenModel, self).__init__()
 
 
        
        
        # -----------------------------
        # Parameters
        # -----------------------------
        self.params = {
            # Calcium flux parameters
            "k_PMCA": 0.4,      # Wang extrusion rate 
            "V_Pmax": 4.5,      # Wang PMCA pump max rate 
            "V_s": 4.5,         # Wang/Lytton SR uptake rate max for SERCA pump between 0.1 and 0.5
            "k_s": 0.1,         # Wang/Lytton SR uptake affinity for SERCA pump between 0.1 and 0.5 
            "k_leak": 0.1,      # Wang SR leak
            "F": 96485.3329,    # Physical Faraday's constant 

            "P": 0.35, # IP3 production proxy by agonist 

            "gamma": 5.5,
            "delta": 0.05,

            "g_Ca": 9.0,        # Wang nS mM^-1 
            "V_m": -50.0,       # Wang mV 
            "k_m": 12.0,        # Wang mV 
            "R": 8314,         # Physical mJ/(mol K)
            "T": 310.0,          # Wang K (37°C) 

            "a_0": 0.05, # Wang
            "a_1": 0.25, # Wang
            "a_2": 1, # Wang

            "k_1": 2000, # Wang,. DeY 
            "k_-1": 260, # Wang, DeY
            "K_1": 0.13, # Wang, DeY - derived 
            "k_2": 1, # Wang, DeY
            "k_-2": 1.05, # Wang, DeY
            "K_2": 1.05, # Wang, DeY - derived
            "k_3": 2000, # Wang, DeY
            "k_-3": 1886, # Wang, DeY
            "K_3": 0.943, # Wang, DeY - derived
            "k_4": 1, # Wang, DeY
            "k_-4": 0.145, # Wang, DeY
            "K_4": 0.145, # Wang, DeY - derived
            "k_5": 100, # Wang, DeY
            "k_-5": 8.2, # Wang, DeY
            "K_5": 0.082, # Wang, DeY - derived

            # Channel gains
            "k_IP3R": 5.55, #Wang
            "k_RyR": 5.0, #Wang 
            "k_ryr0": 0.0072, # Wang and Friel RyR opening rate
            "k_ryr1": 0.334, # Wang and Friel and Shannon RyR closing rate
            "k_ryr2": 0.5, # Wang and Friel and Shannon RyR activation affinity
            "k_ryr3": 38.0, # Wang and Shannon RyR inactivation affinity

            # Voltage parameters
            "c_m": 1.0, # Lata
            "I_stim": 0.1175, # Lata

            # Contraction
            "alpha": 3.0, #Lata - uterine
            "beta": 0.001, #Lata - uterine
            "n_F": 4, # Lata - uterine

            # Functional parameters
            "n": 4, # Wang - Hill coefficient for SERCA channel activation 1, 2 or 4
            "ns": 2, # Wang and Lytton
            "n2": 3, # Wang 3 or 5 

            "E": 5, 
            "k_e1": 0.01,
            "k_e2": 0.2, 
            "k_e_in": 0.9, 
            "h": 4, 
            "EC50": 2000
        }
        self._NumParams = len(self.params)
        

      
    # -----------------------------
    # Flux definitions
    # -----------------------------

    def e_eff(self, p):
        return (1-p["k_e_in"]*p["E"])

    def f_X(self, p):
        return 1 + p["k_e_in"] * (p["E"]**p["h"]) / (p["EC50"]**p["h"] + p["E"]**p["h"])

    def J_in1(self, V, Ca_in, Ca_0, p):
        return p["a_0"]-(p["a_1"]*self.I_Ca(V, Ca_in, Ca_0, p)/(2*p["F"])) * self.e_eff(p) +p["a_2"]*p["P"] 

    def m_inf(self, V, p):
        return 1.0 / (1.0 + np.exp(-(V - p["V_m"]) / p["k_m"]))

    def V_Ca(self, V, Ca_in, Ca_0, p):
        F = p["F"]
        R = p["R"]
        T = p["T"]

        exp_term = np.exp(-2.0 * V * F / (R * T))

        denom = 1.0 - exp_term
        if np.abs(denom) < 1e-8:
            return 1e-8

        return V * (Ca_in - Ca_0 * exp_term) / denom

    def I_Ca(self, V, Ca_in, Ca_0, p):
        m = self.m_inf(V, p)
        Vca = self.V_Ca(V, Ca_in, Ca_0, p)
        return p["g_Ca"] * (m**2) * Vca

    def J_PMCA_Hill(self, Ca_in, p):
        return p["V_Pmax"] * (Ca_in**p["n"]) / (p["k_PMCA"]**p["n"] + Ca_in**p["n"]) # - Ca_0**p["n"]

    def J_SERCA_Hill(self, Ca_in, p):
        return p["V_s"] * (Ca_in**p["ns"]) / (p["k_s"]**p["ns"] + Ca_in**p["ns"]) # - Ca_0**p["n"]

    def J_leak(self, Ca_SR, Ca_in, p):
        return p["k_leak"] * (Ca_SR - Ca_in)

    def J_IP3R_Wang(self, Ca_SR, Ca_in, y_g, p):
        return p["k_IP3R"] * self.P_IP3R(Ca_in, y_g, p) * (Ca_SR - Ca_in)

    def P_IP3R(self, Ca_in, y, p):
        num = p["P"] * Ca_in * (1 - y)
        den = (p["P"] + p["K_1"]) * (Ca_in + p["K_5"])
        return (num / den) ** 3

    def dy_dt(self, y, p, Ca_in):
        f1 = (p["k_-4"] * p["K_2"] * p["K_1"] + p["k_-2"] * p["K_4"] * p["P"]) * Ca_in / (p["K_4"] * p["K_2"] * (p["K_1"] + p["P"]))
        f2 = (p["k_-2"] * p["P"] + p["k_-4"] * p["K_3"]) / (p["K_3"] + p["P"])
        return f1 * (1.0 - y) - f2 * y

    def J_RyR_Wang(self, Ca_SR, Ca_in, p):
        return p["k_RyR"] * self.P_RyR(Ca_in, Ca_SR, p) * (Ca_SR - Ca_in)

    def P_RyR(self,     Ca_in, Ca_SR, p):
        # CICR activation term (cytosolic Ca)
        activation = (
            p["k_ryr0"]
            + (p["k_ryr1"] * Ca_in**3) / (p["k_ryr2"]**3 + Ca_in**3)
        )

        # SR load dependence
        sr_term = Ca_SR**4 / (p["k_ryr3"]**4 + Ca_SR**4)

        return activation * sr_term
        
    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        # Needs to be the number of data time series we have 
        return self.outputs

    def set_outputs(self, Outputs):
        self.outputs  = Outputs

    def set_SimulationTime(self, SimLength):
        self.SimLength  = SimLength
    
    def n_parameters(self):
        return self._NumParams

    def SetNumParams(self, NumberOfParameters ):
        self._NumParams  = NumberOfParameters

    def Setkel_a(self, kel_a):
        self.kel_a  = kel_a 
 

    # -----------------------------
    # ODE system
    # -----------------------------

    def _rhs(self, t, Y, p):
        # _rhs(self, t, Y, CL,V_in,beta, T_0):
        Ca_0, Ca_in, Ca_SR, V, y_g = Y

        # Fluxes
        Jin = p["delta"]*self.J_in1(V, Ca_in, Ca_0, p)
        JPMCA = p["delta"]*self.J_PMCA_Hill(Ca_in, p)
        JSERCA = self.J_SERCA_Hill(Ca_in, p)

        dyg_dt = self.dy_dt(y_g, p, Ca_in)

        Jip3r = self.J_IP3R_Wang(Ca_SR, Ca_in, y_g, p)
        Jryr = self.J_RyR_Wang(Ca_SR, Ca_in, p)
        Jleak = self.J_leak(Ca_SR, Ca_in, p)

        # Calcium dynamics
        dCa0_dt = JPMCA - Jin
        dCain_dt = Jin - JPMCA - JSERCA + Jip3r + Jryr + Jleak
        dCaSR_dt = p["gamma"]*(JSERCA - Jip3r - Jryr - Jleak)

        # Membrane voltage (simple RC model)
        dV_dt = 1/p["c_m"]*(Ca_0 - Ca_in) 

        return [dCa0_dt, dCain_dt, dCaSR_dt, dV_dt, dyg_dt]
    
    # -----------------------------
    # Contraction function
    # -----------------------------
    def contraction(self,Ca_in, p):
        return (p["alpha"] * Ca_in**p["n_F"] / ((p["beta"]*(1+p["k_e2"]*p["E"]))**p["n_F"] + Ca_in**p["n_F"])) * (1-p["k_e1"]*p["E"])
     
    def SetUpSimulation(self):

        print("Not sure if this will be necessary")
        

  

  

    def simulate(self, parameters, times):
        parameters = self.params
        y0 = [1000, 0.112, 24, -60, 0]  # initial conditions for Ca_0, Ca_in, Ca_SR, V, y_g Wang 

        t_span = (0, 100)
        t_eval = np.linspace(*t_span, 1000)

    #  self.y0 = self.GetInitialConditions(T_0 = 36, beta = beta)
    #    

        # Solution = solve_ivp(fun=self._rhs, t_span=[times[0], times[-1]],
        #                 y0=self.y0, t_eval=times, 
        #                 args=( CL, V_in ,beta,  36), method='LSODA' ) #,rtol=1e-10, atol=1e-8) # , atol=1e-1) #  , method='LSODA') # RK45
        GetInitialConditions = self.GetInitialConditions()
        print("Initial conditions from root finding: ", GetInitialConditions)
        sol = solve_ivp(self._rhs, t_span, y0, args=(parameters,), t_eval=t_eval, method='BDF')

        # Extract variables
        Ca_0 = sol.y[0]
        Ca_in = sol.y[1]
        Ca_SR = sol.y[2]
        V = sol.y[3]
        y_g = sol.y[4]

        F = self.contraction(Ca_in, parameters)

        return [Ca_0,Ca_in,Ca_SR, V, y_g, F]
   
 
    def SetDose(self,Dose): 
        print("Can use this to set stimulation")
        self.Dose = Dose 
     
      
  
    def SteadyStateSystem(self,vars, p):
        ( Ca_0, Ca_in, Ca_SR, V, y_g ) = vars  


       
        # Fluxes
        Jin = p["delta"]*self.J_in1(V, Ca_in, Ca_0, p)
        JPMCA = p["delta"]*self.J_PMCA_Hill(Ca_in, p)
        JSERCA = self.J_SERCA_Hill(Ca_in, p)

        dyg_dt = self.dy_dt(y_g, p, Ca_in)

        Jip3r = self.J_IP3R_Wang(Ca_SR, Ca_in, y_g, p)
        Jryr = self.J_RyR_Wang(Ca_SR, Ca_in, p)
        Jleak = self.J_leak(Ca_SR, Ca_in, p)

       
        
       

        # pdb.set_trace()
        eqs = [ 
              # Calcium dynamics
                JPMCA - Jin,
                Jin - JPMCA - JSERCA + Jip3r + Jryr + Jleak,
                p["gamma"]*(JSERCA - Jip3r - Jryr - Jleak),
                #
                # Membrane voltage (simple RC model)
                1/p["c_m"]*(Ca_0 - Ca_in) ,
                self.dy_dt(y_g, p, Ca_in),
          ]
        # pdb.set_trace()
        return eqs
    
 
    
 
 


    # def GetInitialConditions(self):
    
 
    #     InitialGuess = [ -6.16045554e+03, -5.81473909e-03,  1.47287561e-01, 1,1]
 
    #     RootFindingResults = root(self.SteadyStateSystem, InitialGuess, args=(self.params), method='hybr')
 
  
    #     return RootFindingResults.x

    def GetInitialConditions(self, n_restarts=10, verbose=True):
    # """
    # Solve for the steady state of the ODE system by finding the root of
    # SteadyStateSystem. At steady state dX/dt = 0 for all state variables.

    # State vector: [Ca_0, Ca_in, Ca_SR, V, y_g]

    # Analytical constraint from dV/dt = 0:
    #     1/c_m * (Ca_0 - Ca_in) = 0  =>  Ca_0 = Ca_in at steady state.
    # This is used to inform the initial guess but is not enforced explicitly
    # (root-finding solves the full 5D system).

    # Parameters
    # ----------
    # n_restarts : int
    #     Number of random restarts if the primary solve fails.
    # verbose : bool
    #     Print convergence information.

    # Returns
    # -------
    # np.ndarray
    #     Steady-state values [Ca_0, Ca_in, Ca_SR, V, y_g], or None if
    #     no solution is found.
    # """
        p = self.params

        # Physiologically reasonable primary guess, consistent with simulate()
        # Units: Ca in uM, V in mV, y_g dimensionless
        primary_guess = np.array([0.112, 0.112, 24.0, -60.0, 0.5])

        # Physical bounds: Ca concentrations > 0, V in [-120, 60] mV, y_g in [0, 1]
        lower_bounds = [1e-6, 1e-6, 1e-3, -120.0, 0.0]
        upper_bounds = [10.0,  10.0, 500.0,  60.0,  1.0]

        def _residuals(x):
            return self.SteadyStateSystem(x, p)
        import scipy.optimize
        # --- Primary attempt: least_squares with bounds ---
        result = scipy.optimize.least_squares(
            _residuals,
            primary_guess,
            bounds=(lower_bounds, upper_bounds),
            method='trf',
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=10000
        )

        if result.cost < 1e-10:
            if verbose:
                print(f"Steady state found (primary attempt): {result.x}")
                print(f"  Residual norm: {result.cost:.3e}")
            return result.x

        # --- Multi-start fallback ---
        if verbose:
            print(f"Primary attempt did not converge (cost={result.cost:.3e}). "
                f"Attempting {n_restarts} random restarts...")

        best_result = result
        rng = np.random.default_rng(seed=42)

        for i in range(n_restarts):
            # Draw guess uniformly within physical bounds
            guess = rng.uniform(low=lower_bounds, high=upper_bounds)

            res = scipy.optimize.least_squares(
                _residuals,
                guess,
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=10000
            )

            if res.cost < best_result.cost:
                best_result = res

            if best_result.cost < 1e-10:
                if verbose:
                    print(f"Steady state found (restart {i+1}): {best_result.x}")
                    print(f"  Residual norm: {best_result.cost:.3e}")
                return best_result.x

        if verbose:
            print(f"WARNING: No steady state converged to tolerance. "
                f"Best residual norm: {best_result.cost:.3e}")
            print(f"  Best estimate: {best_result.x}")

        return best_result.x
    
        