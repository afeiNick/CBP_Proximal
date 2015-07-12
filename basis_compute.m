function [c, u, v, theta, r] = basis_compute(phi)

phi_half_delta = circshift(phi, [1,0]);
phi_neg_half_delta = circshift(phi, [-1,0]);

first_term = phi - phi_half_delta;
second_term = phi_neg_half_delta - phi_half_delta;

theta = 2 * acos( dot(first_term/norm(first_term), second_term/norm(second_term)) );
r = norm(first_term) / sqrt( 2 * (1 - cos(theta)) );

% compute c, u, v
A_inv = [ 1/(2*(1-cos(theta)))     -cos(theta)/(1-cos(theta))      1/(2*(1-cos(theta))); ...
         -1/(2*r*(1-cos(theta)))    1/(r*(1-cos(theta)))          -1/(2*r*(1-cos(theta))); ...
         -1/(2*r*sin(theta))        0                              1/(2*r*sin(theta))];

phi_polar = A_inv * [phi_neg_half_delta'; phi'; phi_half_delta'];

c = phi_polar(1,:)';
u = phi_polar(2,:)';
v = phi_polar(3,:)';