% project onto the non-convex constraint of the polar approximation
% ADMM
function [ADMM_x1_new, ADMM_z_new, ADMM_u1_new, res] = ...
            ProxPolarADMM_binary(ADMM_z, ADMM_u1, ADMM_phi, ADMM_y, tau, r, theta, noise_std)


N = size(ADMM_z,1);  % the sampling grid size (the coarser one)

lambda1 = 0;
lambda2 = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update x %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ADMM_x1_new = zeros(N,6);
rhs_hat = fft( 1/(2*noise_std^2)*ADMM_y + tau * (ADMM_z - ADMM_u1) );

for k = 1 : N
    phi_hat = reshape(ADMM_phi(k,:),6,6);   
    ADMM_x1_new(k,:) = ((1/(2*noise_std^2)*phi_hat + tau * eye(6))\rhs_hat(k,:)')';
end

ADMM_x1_new = ifft(ADMM_x1_new);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% do projection here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
coeff1 = ADMM_x1_new(:,[1,2,3]) + ADMM_u1(:,[1,2,3]);
coeff2 = ADMM_x1_new(:,[4,5,6]) + ADMM_u1(:,[4,5,6]);

ADMM_z_new(:,[1,2,3]) = ADMM_proj_binary(coeff1 - [lambda1*ones(N,1) zeros(N,2)], r(1), theta(1), 0.95);
ADMM_z_new(:,[4,5,6]) = ADMM_proj_binary(coeff2 - [lambda2*ones(N,1) zeros(N,2)], r(2), theta(2), 0.99);

res = norm(ADMM_z_new - ADMM_z);

ADMM_u1_new = ADMM_u1 + ADMM_x1_new - ADMM_z_new;

