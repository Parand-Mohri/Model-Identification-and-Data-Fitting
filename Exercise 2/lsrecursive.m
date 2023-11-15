function [theta,H,L]=lsrecursive(thetaold,Hold,Lold,phi,y);

% Calculation of one updating step of the least squares estimate of the
% linear regression model associated with y, phi and theta:
%
% y = phi'*theta + e
%
% Input arguments:
% thetaold -- old least squares estimate of parameter vector theta.
% Hold -- old inverse of the matrix Phi'*Phi, where Phi contains all vectors phi.
% Lold -- old sum of squares of residuals e, based on thetaold.
% phi -- new regression vector.
% y -- new datapoint.
% Output arguments:
% theta -- new least squares estimate of parameter vector theta.
% H -- new inverse of the new matrix Phi?*Phi.
% L -- new sum of squares of residuals e, based on theta.
%

g=Hold*phi;
d=1+phi'*g;
H=Hold-1/d*g*g';
e=y-phi'*thetaold;
theta=thetaold+g*e/d;
L=Lold+(e^2)/d;


