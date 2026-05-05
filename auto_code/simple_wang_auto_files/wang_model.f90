
! This file defines the Fortran routines required for AUTO-07P to perform bifurcation
! analysis of our immune model.


! Detailed Breakdown of the Routines:

!     FUNC: Defines the system of differential equations.

!     STPNT: Sets the initial conditions for the variables and the parameters.

!     BCND: Defines the boundary conditions. Since this is an ordinary differential 
!           equation (ODE) problem, there are no boundary conditions, so this is left empty.

!     ICND: Defines any integral constraints. For this problem, there are no integral 
!           constraints, so this routine remains empty as well.

!     FOPT: Defines an optional function that can be used for optimization or additional 
!           output. This is not needed for our system, so we leave it as a placeholder.

!     PVLS: This routine can be used to monitor certain variables or parameters during the 
!           continuation process.

! Additional Notes:

!     par array: This is the parameter array p(1) = rho_P, p(2) = rho_A
!     u array: This represents the state variables:
!              u(1)= J    u(2) = M,   u(3) = A,   u(4) = P

! -------------------------------------------------------------------------------------------------

! FUNC: Defines the system of differential equations.

! Input arguments :
!      ndim   :   Dimension of the algebraic or ODE system 
!      u      :   State variables
!      icp    :   Array indicating the free parameter(s)
!      par    :   Equation parameters

! Values to be returned :
!      f      :   Equation or ODE right hand side values

! Normally unused Jacobian arguments : ijac, dfdu, dfdp (see manual)

subroutine FUNC(ndim, u, icp, par, ijac, f, dfdu, dfdp)

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ndim, ijac, icp(*)
    DOUBLE PRECISION, INTENT(IN) :: u(ndim), par(*)
    DOUBLE PRECISION, INTENT(OUT) :: f(ndim)
    DOUBLE PRECISION, INTENT(INOUT) :: dfdu(ndim,ndim), dfdp(ndim,*)

    ! Fixed parameters
	
	! Calcium flux parameters
	DOUBLE PRECISION :: Ca_0 = 2.0 		!Placeholder calcium outside the cell
    DOUBLE PRECISION :: k_PMCA = 0.4      ! Wang extrusion rate 
   	DOUBLE PRECISION :: V_Pmax = 4.5      ! Wang PMCA pump max rate 
	DOUBLE PRECISION :: V_s = 4.5         ! Wang/Lytton SR uptake rate max for SERCA pump between 0.1 and 0.5
	DOUBLE PRECISION :: k_s = 0.1         ! Wang/Lytton SR uptake affinity for SERCA pump between 0.1 and 0.5 
	DOUBLE PRECISION :: k_leak = 0.1      ! Wang SR leak
	DOUBLE PRECISION :: Faraday = 96485.3329    ! Physical Faraday's constant 

	DOUBLE PRECISION ::  P = 0.0 ! IP3 production proxy by agonist

	DOUBLE PRECISION ::  gamma = 5.5
	DOUBLE PRECISION ::  delta = 0.05 ! Initial closed cell model 0.05 normally

	DOUBLE PRECISION ::  g_Ca = 9.0        ! Wang nS mM^-1 
	DOUBLE PRECISION ::  V_m = -50.0       ! Wang mV Resting membrane potenital
	DOUBLE PRECISION ::  k_m = 12.0        ! Wang mV 
	DOUBLE PRECISION ::  R = 8314         ! Physical mJ/(mol K)
	DOUBLE PRECISION ::  T = 310.0          ! Wang K (37°C) 
	
	!DOUBLE PRECISION ::  V = -40.0 !Current membrane potential

	DOUBLE PRECISION ::  a_0 = 0.05 ! Wang
	DOUBLE PRECISION ::  a_1 = 0.25 ! Wang
	DOUBLE PRECISION ::  a_2 = 1 ! Wang

	!DOUBLE PRECISION ::  k_1 = 2000, ! Wang,. DeY 
	!DOUBLE PRECISION ::  k_-1 = 260, ! Wang, DeY
	!DOUBLE PRECISION ::  K_1 = 0.13, ! Wang, DeY - derived 
	!DOUBLE PRECISION ::  k_2 = 1, ! Wang, DeY
	!DOUBLE PRECISION ::  k_-2 = 1.05, ! Wang, DeY
	!DOUBLE PRECISION ::  K_2 = 1.05, ! Wang, DeY - derived
	!DOUBLE PRECISION ::  k_3 = 2000, ! Wang, DeY
	!DOUBLE PRECISION ::  k_-3 = 1886, ! Wang, DeY
	!DOUBLE PRECISION ::  K_3 = 0.943, ! Wang, DeY - derived
	!DOUBLE PRECISION ::  k_4 = 1, ! Wang, DeY
	!DOUBLE PRECISION ::  k_-4 = 0.145, ! Wang, DeY
	!DOUBLE PRECISION ::  K_4 = 0.145, ! Wang, DeY - derived
	!DOUBLE PRECISION ::  k_5 = 100, ! Wang, DeY
	!DOUBLE PRECISION ::  k_-5 = 8.2, ! Wang, DeY
	!DOUBLE PRECISION ::  K_5 = 0.082, ! Wang, DeY - derived

	! Channel gains
	DOUBLE PRECISION ::  k_IP3R = 5.55 !Wang
	DOUBLE PRECISION ::  k_RyR = 5.0 !Wang 
	DOUBLE PRECISION ::  k_ryr0 = 0.0072 ! Wang and Friel RyR opening rate
	DOUBLE PRECISION ::  k_ryr1 = 0.334 ! Wang and Friel and Shannon RyR closing rate
	DOUBLE PRECISION ::  k_ryr2 = 0.5 ! Wang and Friel and Shannon RyR activation affinity
	DOUBLE PRECISION ::  k_ryr3 = 38.0 ! Wang and Shannon RyR inactivation affinity

	! Functional parameters
	DOUBLE PRECISION ::  n = 4 ! Wang - Hill coefficient for SERCA channel activation 1, 2 or 4
	DOUBLE PRECISION ::  ns = 2 ! Wang and Lytton
	DOUBLE PRECISION ::  n2 = 3 ! Wang 3 or 5 
	
	!Local variables to help with calculations
	DOUBLE PRECISION :: exp_term_den, Vca, m, I_Ca, P_RyR, activation, sr_term,exp_term
	DOUBLE PRECISION :: Jin, Jpmca, Jserca,Jipr, Jryr, Jleak 

    ! Extract the state variables and continuation parameters
    DOUBLE PRECISION Ca_in, Ca_sr ! State variables
    DOUBLE PRECISION V ! Parameters (of interest) - Wang used total calcium, I might prefer to use V

    Ca_in = u(1)
    Ca_SR = u(2)

	V=par(1)
    !Flux terms
	
	!Jin term, calculating voltage gated information.
	exp_term = EXP(-2.0 * V * Faraday / (R * T))
    exp_term_den = 1.0 - exp_term
	IF(exp_term_den.LT.1e-8)THEN !ARC check this
		Vca = 1e-8
	ELSE
		Vca = V * (Ca_in - Ca_0 * exp_term) / exp_term_den
	ENDIF
	m = 1.0 / (1.0 + EXP(-(V - V_m) /k_m))
	I_Ca = g_Ca * m**2 * Vca
    Jin = a_0-a_1*I_Ca/(2*Faraday) +a_2*P
	
	!Jpmca
    Jpmca = V_Pmax * (Ca_in**n) / (k_PMCA**n + Ca_in**n)
	!Jserca
    Jserca = V_s * (Ca_in**ns) / (k_s**ns + Ca_in**ns) 
	!ip3r = 0.0 [ASSUMPTION p=0]
    Jipr = 0.0
	!Jryr
	!CICR activation term (cytosolic Ca)
	activation = (k_ryr0+ (k_ryr1 * Ca_in**3) / (k_ryr2**3 + Ca_in**3))
	!SR load dependence
	sr_term = Ca_SR**4 / (k_ryr3**4 + Ca_SR**4)
	P_RyR = activation * sr_term
    Jryr = k_RyR * P_RyR * (Ca_SR - Ca_in)
	!Jleak
    Jleak = k_leak * (Ca_SR - Ca_in)


    ! System equations ---------------------------------------------------------------------

    f(1) = delta*(Jin-Jpmca)-Jserca+Jipr+Jryr+Jleak! dCindt
    f(2) = gamma*(Jserca-Jipr-Jryr-Jleak) ! dCsrdt


    IF(ijac.EQ.0)RETURN
    
    ! The jacobian -------------------------------------------------------------------------
    
	!Is currently too hard.
    ! dJ/dt = -k1*A_cyto*J + k2*M + konJ*J_cyto - koffJ*J - kJP*P*J
    dfdu(1,1) = 0!-k1*A_cyto*J - konJ*psi - koffJ - kJP*P ! dJ
    dfdu(1,2) = 0!k1*psi*J + k2 - konJ*psi ! dM
    dfdu(1,3) = 0!k1*psi*J ! dA
    dfdu(1,4) = 0!- kJP*J ! dP
    ! dM/dt = k1*A_cyto*J - k2*M - koffM*M - kMP*P*M
    dfdu(2,1) = 0!k1*A_cyto ! dJ
    dfdu(2,2) = 0!-k1*psi*J - k2 - koffM - kMP*P ! dM
    dfdu(2,3) = 0!-k1*psi*J ! dA
    dfdu(2,4) = 0!- kMP*M ! dP
    ! dA/dt = k2*M + konA*A_cyto - koffA*A - kAP*P*A
    dfdu(3,1) = 0 ! dJ
    dfdu(3,2) = 0!k2 - konA*psi ! dM
    dfdu(3,3) = 0!-konA*psi - koffA - kAP*P ! dA
    dfdu(3,4) = 0!- kAP*A ! dP
    ! dP/dt = konP*P_cyto - koffP*P - kPA*(A+M)*(A+M)*P
    dfdu(4,1) =0! 0 ! dJ
    dfdu(4,2) =0! - 2.0*kPA*(A+M)*P ! dM
    dfdu(4,3) =0! - 2.0*kPA*(A+M)*P ! dA
    dfdu(4,4) =0! -konP*psi - koffP - kPA*(A+M)*(A+M) ! dP



end subroutine FUNC

! -------------------------------------------------------------------------------------------------

! STPNT: Sets the initial conditions for the variables and the parameters.

! Input arguments :
!      ndim   :   Dimension of the algebraic or ODE system 

! Values to be returned :
!      u      :   A starting solution vector
!      par    :   The corresponding equation-parameter values

! Note : For time- or space-dependent solutions this subroutine has
!        the scalar input parameter t contains the varying time or space
!        variable value.

subroutine STPNT(ndim, u, par, t)
    
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ndim
    DOUBLE PRECISION, INTENT(INOUT) :: u(ndim),par(*)
    DOUBLE PRECISION, INTENT(IN) :: t

  
    ! Anterior dominant ss
    par(1) = -60.0 ! Initial value for  parmeter of interst
    u(1) = 0.16097807 ! Cin SS Value
    u(2) = 27.41010233 ! C_SR SS value



end subroutine STPNT

! -------------------------------------------------------------------------------------------------

!     BCND: Defines the boundary conditions. Since this is an ordinary differential 
!           equation (ODE) problem, there are no boundary conditions, so this is left empty.

!     ICND: Defines any integral constraints. For this problem, there are no integral 
!           constraints, so this routine remains empty as well.

!     FOPT: Defines an optional function that can be used for optimization or additional 
!           output. This is not needed for our system, so we leave it as a placeholder.

!     PVLS: This routine can be used to monitor certain variables or parameters during the 
!           continuation process.

subroutine BCND(ndim, par, t, u0, u1, ijac, fb, dbc)
end subroutine BCND
  
subroutine ICND(ndim, par, t, u, udot, ijac, fi, dfi)
end subroutine ICND
  
subroutine FOPT(ndim, u, icp, par, ijac, fs, dfdu, dfdp)
end subroutine FOPT
  
subroutine PVLS(ndim, u, par)
end subroutine PVLS
  