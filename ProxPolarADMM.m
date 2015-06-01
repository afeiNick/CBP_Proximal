% project onto the non-convex constraint of the polar approximation
% ADMM
function [ADMM_x_new, ADMM_z_new, ADMM_u_new] = ProxPolarADMM(ADMM_x, ADMM_z, ADMM_u, ADMM_phi, ADMM_y, tau, lambda, r, theta)
% ADMM_y = PolarS(y)


n = size(ADMM_x,1);  % the sampling grid size
ADMM_x_new = zeros(n,3);
rhs_hat = fft( ADMM_y + tau * (ADMM_z - ADMM_u) );

for k = 1 : n
    %ADMM_phi = [ADMM_phi_11 ADMM_phi_21 ADMM_phi_31 ADMM_phi_12 ADMM_phi_22 ADMM_phi_32 ADMM_phi_13 ADMM_phi_23 ADMM_phi_33];
    phi_hat = reshape(ADMM_phi(k,:),3,3);
    
    % fft applies to each column of the matrix, but here we want to
    % apply to the rows; u_hat is now a column vector
    ADMM_x_new(k,:) = ((phi_hat + tau * eye(3))\rhs_hat(k,:)')';
end

ADMM_x_new = ifft(ADMM_x_new);


x = ADMM_x_new(:,1) + ADMM_u(:,1) - lambda;
y = ADMM_x_new(:,2) + ADMM_u(:,2) - lambda;
z = ADMM_x_new(:,3) + ADMM_u(:,3) - lambda;


% project onto the non-convex contraint set
for i = 1 : n
    
    if (abs(atan(z(i)/y(i))) <= theta)  % if we are within the sector
        theta_proj = atan(z(i)/y(i));
    elseif (z(i) > 0)                   % otherwise the projection will be at one of the endpoints of the arc
        theta_proj = theta;
    elseif (z(i) < 0)
        theta_proj = -theta;
    elseif (z(i) == 0)                  % when z(i) = 0, we pick either endpoint at random
        if (y(i) == 0)
            theta_proj = unifrnd(-theta,theta);
        else
            if rand <= 0.5
                theta_proj = theta;
            else
                theta_proj = -theta;
            end
        end
    end
    % now we project onto the line specified by [1/r cos(theta_proj)
    % sin(theta_proj)]
    unit_vec = [1/r cos(theta_proj) sin(theta_proj)]'/norm([1/r cos(theta_proj) sin(theta_proj)]');
    projection_length = dot([x(i) y(i) z(i)]', unit_vec);
    x(i) = projection_length * unit_vec(1);
    y(i) = projection_length * unit_vec(2);
    z(i) = projection_length * unit_vec(3);
    
    if x(i) <= 0
        x(i) = 0;
        y(i) = 0;
        z(i) = 0;
    end
  
end

ADMM_z_new(:,1) = x;
ADMM_z_new(:,2) = y;
ADMM_z_new(:,3) = z;

ADMM_u_new = ADMM_u + ADMM_x_new - ADMM_z_new;

