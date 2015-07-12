% This code does CBP stuff
% Yifei Sun
% 06/25/2015

n_cells = 2;

n_iter = 1;
QUIET  = 0;
tau    = 0.01;        % initial tau

lambda = 0;

% noise level
noise_std = 0.2;

% specify the waveforms
phi1 = @(t)2*100*t.*exp(-(100*t).^2);  % it looks this waveform is easier to identify
%phi1 = @(t)1.5*(exp(-(128*t).^4 / 16) - exp(-(128*t).^2));
% phi1 = @(t)0.5*exp(-t.^2/(2*0.005^2))
phi2 = @(t)(exp(-(100*t).^4 / 16) - exp(-(100*t).^2));
% phi2 = @(t)256*t.*exp(-(128*t).^2);
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


% Forward-backward splitting
coeff = zeros(N, n_cells*3);
gradF = @(u)1/(2*noise_std^2)*PolarS(Polar(u)-y);

for i=1:n_iter
    
    % forward step
    coeff = coeff - tau * gradF(coeff);
    coeff1 = coeff(:,[1,2,3]) - [lambda*ones(N,1), zeros(N,2)];
    coeff2 = coeff(:,[4,5,6]) - [lambda*ones(N,1), zeros(N,2)];
   
    
    % backward projection
    coeff1 = ADMM_proj_binary(coeff1, r(1), theta(1), 0.1);
    coeff2 = ADMM_proj_binary(coeff2, r(2), theta(2), 0.3);
    
    coeff = [coeff1, coeff2];
    % computes certain things
    history.objval(i)  = norm(y_clean - Polar(coeff))^2;   % the data-fidelity term
        
    if ~QUIET
        if mod(i,1)==0
        fprintf('%3d\t%10.2f\n', i, history.objval(i));
        end
    end

end

figure(2)
y_approx = Polar(coeff);
plot(t1,[y_clean y_approx],'LineWidth', 2)
legend('true solution', 'approximated solution')

J1 = find(coeff(:,1)>1e-3);
J2 = find(coeff(:,4)>1e-3);
plot_grid = (0:N-1)'/N;

figure(3)
subplot(2,1,1); hold on;
mystem(x1(I1), a1(I1), 'k', msB, lw); % initial spikes
mystem(plot_grid(J1) + Delta/(2*theta(1)) * atan(coeff(J1,3)./coeff(J1,2)), coeff(J1,1), 'r', 20, 1.5);  % recovered spikes
axis([0 1 0 1]);
box on;

subplot(2,1,2);
mystem(x2(I2), a2(I2), 'k', msB, lw); % initial spikes
hold on;
mystem(plot_grid(J2) + Delta/(2*theta(2)) * atan(coeff(J2,6)./coeff(J2,5)), coeff(J2,4), 'r', 20, 1.5);  % recovered spikes
axis([0 1 0 1]);
box on;
