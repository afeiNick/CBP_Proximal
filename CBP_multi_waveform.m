% This code does CBP stuff
% Yifei Sun
% 06/25/2015

n_cells = 2;

n_iter = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
QUIET    = 0;
mu      = 10;   % used for varying penalty parameter
increase_factor = 2;
tau = 1.;        % initial tau

% noise level
noise_std = 0.2;

% specify the waveforms
phi1 = @(t)2*100*t.*exp(-(100*t).^2);  % it looks this waveform is easier to identify
% phi1 = @(t)1.5*(exp(-(128*t).^4 / 16) - exp(-(128*t).^2));
% phi1 = @(t)0.5*exp(-t.^2/(2*0.01^2));
phi2 = @(t)(exp(-(100*t).^4 / 16) - exp(-(100*t).^2));
% phi2 = @(t)256*t.*exp(-(128*t).^2); In fir
% phi2  = @(t)0.5*exp(-t.^2/(2*0.01^2));


% grid size
N = 64;
Delta = 1/N;
rho = 64;
P = N*rho;      % the number of observation points

% functions for doing conv and its adjoint
% if h is a matrix rather than a column vector, we do convolution for each
% column of h with x
convol  = @(x,h)real(ifft(fft( repmat(x,1,size(h,2)) ).*fft(h)));
convolS = @(x,h)real(ifft(fft( repmat(x,1,size(h,2)) ).*conj(fft(h))));

% specify the domain
t = [0:P/2, -P/2+1:-1]' / P;  % [0,1,...,P/2, -P/2+1, -P/2+2, ... , -1]; this is just because we have a periodic boundary condition
t1 = (0:P-1)'/P;   % this t1 is for plotting, not computing

[c1_d, u1_d, v1_d, theta1, r1] = basis_compute(phi1(t));
[c2_d, u2_d, v2_d, theta2, r2] = basis_compute(phi2(t));

basis = [c1_d u1_d v1_d c2_d u2_d v2_d];
theta = [theta1 theta2];
r = [r1 r2];


C1 = @(x)convol(upsample(x,rho),c1_d);
U1 = @(x)convol(upsample(x,rho),u1_d);
V1 = @(x)convol(upsample(x,rho),v1_d);

C2 = @(x)convol(upsample(x,rho),c2_d);
U2 = @(x)convol(upsample(x,rho),u2_d);
V2 = @(x)convol(upsample(x,rho),v2_d);

Polar  = @(u)C1(u(:,1)) + U1(u(:,2)) + V1(u(:,3)) + C2(u(:,4)) + U2(u(:,5)) + V2(u(:,6));  % this gives the approximation
% PolarS = @(y)[C1_S(y) U1_S(y) V1_S(y) C2_S(y) U2_S(y) V2_S(y)];
PolarS = @(y)downsample(convolS(y, basis),rho);


% the phi matrix in ADMM
ADMM_phi_1 = fft(downsample(convolS(c1_d, basis), rho)); % each is a column of length n
ADMM_phi_2 = fft(downsample(convolS(u1_d, basis), rho)); 
ADMM_phi_3 = fft(downsample(convolS(v1_d, basis), rho));
ADMM_phi_4 = fft(downsample(convolS(c2_d, basis), rho));
ADMM_phi_5 = fft(downsample(convolS(u2_d, basis), rho));
ADMM_phi_6 = fft(downsample(convolS(v2_d, basis), rho));

ADMM_phi = [ADMM_phi_1 ADMM_phi_2 ADMM_phi_3 ADMM_phi_4 ADMM_phi_5 ADMM_phi_6];

kappa = 0.9;

% three spikes
% I1 = [15 20 55];   % set the location of the spikes of waveform 1
I1 = [10 20 40 54];
a1 = zeros(N,1); a1(I1) = [1. 1. 1. 1.];   % choose the amplitude

% I2 = [14 23 54];
I2 = [5 31 45 56];
a2 = zeros(N,1); a2(I2) = [1. 1. 1. 1.];   % choose the amplitude

% add some random shifts off the grid
d1 = zeros(N,1); d1(I1) = [-.5 .2 -.1 .3] * kappa;
d2 = zeros(N,1); d2(I2) = [-.3 -.2 -.5 -.1] * kappa;


% two spikes
% I1 = [14 50];   % set the location of the spikes of waveform 1
% a1 = zeros(N,1); a1(I1) = [1. 1.];   % choose the amplitude
% I2 = [10 47];
% a2 = zeros(N,1); a2(I2) = [1. 1.];   % choose the amplitude
% 
% d1 = zeros(N,1); d1(I1) = [.5 -.2] * kappa;
% d2 = zeros(N,1); d2(I2) = [-.3 -.1] * kappa;


% true spike locations
x1 = (0:N-1)'/N + d1*Delta/2;
x2 = (0:N-1)'/N + d2*Delta/2;

% generate observation y(t)
y_clean = zeros(P,1);    
for i = 1 : N
    T1 = t - x1(i); T1 = mod(T1,1); T1(T1>0.5) = T1(T1>0.5)-1;
    T2 = t - x2(i); T2 = mod(T2,1); T2(T2>0.5) = T2(T2>0.5)-1;
    y_clean = y_clean + a1(i) * phi1( T1 ) +  a2(i) * phi2( T2 );
end

% add noise to the observation
y = y_clean + normrnd(0,noise_std,[P,1]);

% plotting tools
lw = 2; msB = 30;
mystem = @(x,y, col, msB, lw)stem(x, y, [col '.--'], 'MarkerSize', msB, 'LineWidth', lw);

figure(1)
subplot(2,1,1); hold on;
plot(t1, y_clean, 'LineWidth', lw); axis tight; title('True Observations');
subplot(2,1,2);
plot(t1, y, 'LineWidth', lw); axis tight; title('Noisy Observations');


% ADMM iteration
ADMM_z = zeros(N,3*n_cells);
ADMM_u1 = zeros(N,3*n_cells);
ADMM_u2 = zeros(N,3*n_cells); % only needed for the consensus algo


for i=1:n_iter
%     [ADMM_z, ADMM_u1, ADMM_u2] = ...
%        ProxPolarADMM_consensus(ADMM_z, ADMM_u1, ADMM_u2, ADMM_phi, PolarS(y), 1500, r, theta, noise_std);

%     [ADMM_z, ADMM_u1, ADMM_u2] = ...
%         ProxPolarADMM_consensus_another(ADMM_z, ADMM_u1, ADMM_u2, ADMM_phi, PolarS(y), 20, r, theta, noise_std);

    [ADMM_x1, ADMM_z, ADMM_u1, res] = ...
        ProxPolarADMM_binary(ADMM_z, ADMM_u1, ADMM_phi, PolarS(y), tau, r, theta, noise_std);

    
    % computes certain things
    history.objval(i)  = norm(y_clean - Polar(ADMM_z))^2;   % the data-fidelity term
    
    history.r_norm(i)  = norm(ADMM_x1 - ADMM_z);
    history.s_norm(i)  = res;

    history.eps_pri(i) = sqrt(N)*ABSTOL + RELTOL*max(norm(ADMM_x1), norm(-ADMM_z));
    history.eps_dual(i)= sqrt(N)*ABSTOL + RELTOL*norm(tau*ADMM_u1);
        
    if ~QUIET
        if mod(i,1)==0
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', i, ...
            history.r_norm(i), history.eps_pri(i), ...
            history.s_norm(i), history.eps_dual(i), history.objval(i));
        end
    end

    if (history.r_norm(i) < history.eps_pri(i) && history.s_norm(i) < history.eps_dual(i))
         break;
    end
    
%     if (history.r_norm(i) > mu * history.s_norm(i) || history.eps_pri(i) > mu * history.eps_dual(i))
%         tau = increase_factor * tau;
%         ADMM_u1 = ADMM_u1 / increase_factor;
%     elseif (history.s_norm(i) > mu * history.r_norm(i) || history.eps_dual(i) > mu * history.eps_pri(i))   
%         tau = tau / increase_factor;
%         ADMM_u1 = increase_factor * ADMM_u1;
%     end

    if (history.r_norm(i) > mu * history.s_norm(i))
        tau = increase_factor * tau;
        ADMM_u1 = ADMM_u1 / increase_factor;
    elseif (history.s_norm(i) > mu * history.r_norm(i))   
        tau = tau / increase_factor;
        ADMM_u1 = increase_factor * ADMM_u1;
    end
    

end

figure(2)
y_approx = Polar(ADMM_z);
plot(t1,[y_clean y_approx],'LineWidth', 2)
legend('true solution', 'approximated solution')

J1 = find(ADMM_z(:,1)>1e-3);
J2 = find(ADMM_z(:,4)>1e-3);
plot_grid = (0:N-1)'/N;

figure(3)
subplot(2,1,1); hold on;
mystem(x1(I1), a1(I1), 'k', msB, lw); % initial spikes
mystem(plot_grid(J1) + Delta/(2*theta(1)) * atan(ADMM_z(J1,3)./ADMM_z(J1,2)), ADMM_z(J1,1), 'r', 20, 1.5);  % recovered spikes
axis([0 1 0 1]);
box on;

subplot(2,1,2);
mystem(x2(I2), a2(I2), 'k', msB, lw); % initial spikes
hold on;
mystem(plot_grid(J2) + Delta/(2*theta(2)) * atan(ADMM_z(J2,6)./ADMM_z(J2,5)), ADMM_z(J2,4), 'r', 20, 1.5);  % recovered spikes
axis([0 1 0 1]);
box on;
