clc;
clear;

% Initial guess
x0 = [100; 50; 25];

% Solver options
options = optimset( ...
    'Display','iter', ...
    'TolFun',1e-10, ...
    'TolX',1e-10);

% Solve steady state
[x_ss, fval, exitflag] = fsolve(@steadystates, x0, options);

% Display results
disp('Steady state:')
disp(x_ss)

disp('Residuals:')
disp(fval)

disp('Exit flag:')
disp(exitflag)