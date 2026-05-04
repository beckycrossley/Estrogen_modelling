function F = steadystates(x)

% Unknown variables
C_0  = x(1);
C_in = x(2);
C_SR = x(3);

% -----------------------------
% Parameters
% -----------------------------
p       = 0.35;

V_p     = 4.5;
k_p     = 0.4;
n       = 4;

a_0     = 0.05;
a_1     = 0.25;
a_2     = 1.0;

V_e     = 4.5;
k_e     = 0.1;

k_IP3R  = 5.55;
k_RYR   = 5.0;

J_er    = 0.1;

gCa     = 9.0;

V_m     = -50;
k_m     = 12;

V       = -60;

Fconst  = 96485.3329;
R       = 8314;
T       = 310;

K_1     = 0.13;
K_5     = 0.082;

y       = 0.1;

% -----------------------------
% Auxiliary variables
% -----------------------------

m = 1/(1 + exp(-(V - V_m)/k_m));

exp_term = exp(-2*V*Fconst/(R*T));

V_Ca = V*(C_in - C_0*exp_term)/(1 - exp_term);

I_Ca = gCa*m^2*V_Ca;

P_RYR = C_in^n/(k_p^n + C_in^n);

P_IP3R = (p*C_in*(1-y)/((p + K_1)*(C_in + K_5)))^3;

% -----------------------------
% Fluxes
% -----------------------------

J_PMCA = V_p*C_in^n/(k_p^n + C_in^n);

J_in = a_0 - a_1*I_Ca/(2*Fconst) + a_2*p;

J_SERCA = V_e*C_in^2/(k_e^2 + C_in^2);

J_IP3R = k_IP3R*P_IP3R*(C_SR - C_in);

J_RYR = k_RYR*P_RYR*(C_SR - C_in);

J_leak = J_er*(C_SR - C_in);

% -----------------------------
% Steady-state equations
% -----------------------------

F = zeros(3,1);

F(1) = J_PMCA - J_in;

F(2) = J_in - J_PMCA - J_SERCA + J_IP3R + J_RYR + J_leak;

F(3) = J_SERCA - J_IP3R - J_RYR - J_leak;

end
