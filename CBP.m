% This code does CBP stuff
% Yifei Sun
% 05/03/2015


% specify the wave form
sigma = .01;
phi  = @(t)exp(-t.^2/(2*sigma^2));
phi1 = @(t)-t/(sigma^2).*exp(-t.^2/(2*sigma^2));


N = 64;
Delta = 1/N;

rho = 64;
P = N*rho;      % the number of observation points is 32*64

% functions for doing conv and its adjoint
convol  = @(x,h)real(ifft(fft(x).*fft(h)));
convolS = @(x,h)real(ifft(fft(x).*conj(fft(h))));


t = [0:P/2, -P/2+1:-1]' / P;  % [0,1,...,P/2, -P/2+1, -P/2+2, ... , -1]; this is just because we have a periodic boundary condition
t1 = (0:P-1)'/P;

% initialize the Gaussian kernels
phi_d = phi(t);  
phi1_d = phi1(t);

% compute theta and r at every location
phi_half_delta = circshift(phi_d, [1,0]);
phi_neg_half_delta = circshift(phi_d, [-1,0]);

first_term = phi_d - phi_half_delta;
second_term = phi_neg_half_delta - phi_half_delta;

theta = 2 * acos( dot(first_term/norm(first_term), second_term/norm(second_term)) );
r = norm(first_term) / sqrt( 2 * (1 - cos(theta)) );

% compute c, u, v
A_inv = [ 1/(2*(1-cos(theta)))     -cos(theta)/(1-cos(theta))      1/(2*(1-cos(theta)));...
         -1/(2*r*(1-cos(theta)))    1/(r*(1-cos(theta)))            -1/(2*r*(1-cos(theta)));...
         -1/(2*r*sin(theta))        0                                  1/(2*r*sin(theta))];
phi_polar = A_inv * [phi_neg_half_delta'; phi_d'; phi_half_delta'];
c_d = phi_polar(1,:)';   % why can't I multiply c by r to scale x -> x/r?
u_d = phi_polar(2,:)';
v_d = phi_polar(3,:)';

% we have to rearrange c_d, u_d, v_d by switching from t1 to t

 

Phi  = @(x)convol(upsample(x,rho),phi_d);
PhiS = @(x)downsample(convolS(x,phi_d),rho);
Psi  = @(s)convol(upsample(s,rho),phi1_d)*Delta/2;
PsiS = @(s)downsample(convolS(s,phi1_d),rho)*Delta/2;
Gamma  = @(u)Phi(u(:,1)) - Psi(u(:,2));
GammaS = @(y)[PhiS(y), -PsiS(y)];


C = @(x)convol(upsample(x,rho),c_d);
C_S = @(x)downsample(convolS(x,c_d),rho);
U = @(x)convol(upsample(x,rho),u_d);
U_S = @(x)downsample(convolS(x,u_d),rho);
V = @(x)convol(upsample(x,rho),v_d);
V_S = @(x)downsample(convolS(x,v_d),rho);
Polar  = @(u)C(u(:,1)) + U(u(:,2)) + V(u(:,3));
PolarS = @(y)[C_S(y) U_S(y) V_S(y)];


% for ADMM
% the phi matrix
ADMM_phi_11 = fft(downsample(convolS(c_d, c_d), rho));  % each is a n column matrix
ADMM_phi_12 = fft(downsample(convolS(u_d, c_d), rho));
ADMM_phi_13 = fft(downsample(convolS(v_d, c_d), rho));

ADMM_phi_21 = fft(downsample(convolS(c_d, u_d), rho));
ADMM_phi_22 = fft(downsample(convolS(u_d, u_d), rho));
ADMM_phi_23 = fft(downsample(convolS(v_d, u_d), rho));

ADMM_phi_31 = fft(downsample(convolS(c_d, v_d), rho));
ADMM_phi_32 = fft(downsample(convolS(u_d, v_d), rho));
ADMM_phi_33 = fft(downsample(convolS(v_d, v_d), rho));

ADMM_phi = [ADMM_phi_11 ADMM_phi_21 ADMM_phi_31 ADMM_phi_12 ADMM_phi_22 ADMM_phi_32 ADMM_phi_13 ADMM_phi_23 ADMM_phi_33];



k = 3; % number of spikes
kappa = .9;
I = round( N/(2*k):N/k:N );   % set the location of the spikes
a0 = zeros(N,1); a0(I) = [.6 1 .8];   % choose the amplitude
d0 = zeros(N,1); d0(I) = [-.2 1 -.7] * kappa;
b0 = d0.*a0;
x0 = (0:N-1)'/N + d0*Delta/2;

y = zeros(P,1) + normrnd(0,1/inf,[P,1]);
for i=1:length(x0)
    T = t-x0(i); T = mod(T,1); T(T>.5) = T(T>.5)-1;
    y = y + a0(i) * phi( T );
end

% true signal and Taylor approximation
y0 = Phi(a0);
y1 = Gamma([a0 b0]);

figure(1)
lw = 2; msB = 30;
mystem = @(x,y, col)stem(x, y, [col '.--'], 'MarkerSize', msB, 'LineWidth', lw);
subplot(3,1,1); hold on;
mystem(x0(I), a0(I), 'k'); box on;
plot(t1, y, 'LineWidth', lw); axis tight; title('Observations');
subplot(3,1,2);
plot(t1, [y-y0 y-y1], 'LineWidth', lw); axis tight; title('0th order');
legend('0th order', '1st order');
subplot(3,1,3);
plot(t1, y-y1, 'g', 'LineWidth', lw); axis tight; title('1st order');


% project onto the convex constraint of the Taylor approximation
R  = @(u)u(:,2)+1i*u(:,1);
Ri = @(v)[imag(v), real(v)];
ProjOct = @(v)max(real(v),0) + 1i*max(imag(v),0);
ProjC = @(u)Ri(exp(1i*pi/4)*ProjOct(exp(-1i*pi/4)*R(u)));
ProxJ = @(w,lambda)ProjC( w-[lambda*ones(size(w,1),1) zeros(size(w,1),1)] );

% define CBP polar projection using the ProjPolar function
ProxPolar = @(w,lambda)ProjPolar( w-[lambda*ones(size(w,1),1) zeros(size(w,1),1) zeros(size(w,1),1)], r, theta);


% the following is for determinating the step size for Taylor
A = @(u)GammaS(Gamma(u));
% A_p = @(u)PolarS(Polar(u)); % polar L^*L

u = randn([N 2]);
u = u/norm(u(:));
%polar
u_p = randn([N 3]);
u_p = u_p/norm(u_p(:));

% power iteration for finding the largest eigenvalue of L^*L
e = [];
for i=1:15
    v = A(u);
    e(end+1) = sum(u(:).*v(:));
    u = v/norm(v(:));
end
L = e(end);
% figure(2)
% clf; plot(e, 'LineWidth', 2); axis tight;

lambda = 0.;
F = @(u)1/2*norm(y-Gamma(u), 'fro')^2;
gradF = @(u)GammaS(Gamma(u)-y);

% polar
F_p = @(u)1/2*norm(y-Polar(u), 'fro')^2;
gradF_p = @(u)PolarS(Polar(u)-y);


E = @(u)F(u) + lambda*norm(u(:,1),1);

R = [];
u = zeros(N,2);
niter = 2000;
damp = .1;   % make damp < 2 so the iteration converges for sure
tau = damp/L;
for i=1:niter
    u = ProxJ( u - tau * gradF(u), lambda*tau );
    R(end+1) = E(u); 
end
sel = 1:niter/4;
% figure(3)
% plot(sel, log(R(sel)/min(R)-1), '-', 'LineWidth', 2); axis tight;

a = u(:,1);
b = u(:,2);
delta = Delta/2 * b./a; delta(a<1e-9) = 0;
x = (0:N-1)'/N + delta;


J = find(a>1e-3);
t = (0:N-1)'/N;
s = (0:P-1)'/P;
figure(4)
clf; hold on;
plot(s, y, 'LineWidth', 2);
mystem(x0(I), a0(I), 'k'); % initial spikes
mystem(t(J) + delta(J), a(J), 'r');  % recovered spikes
axis([0 1 0 1]);
box on;

% figure(5)
% subplot(2,1,1);
% hold on;
% mystem(1:N, a0, 'k');
% plot(a, 'r.', 'MarkerSize', 20);
% axis tight; box on; title('a');
% subplot(2,1,2);
% hold on;
% mystem(1:N, b0, 'k');
% plot(b, 'r.', 'MarkerSize', 20);
% axis tight; box on; title('b');

figure(6)  % comparing the original and approximated observations
y2 = Gamma([a b]);
plot(t1,[y y2])

% polar proximal iteration
niter = 2000;
damp = .1;   % here we attempt to use the same step size as in Taylor, just to see...
tau = damp/L;
for i=1:niter
    u_p = ProxPolar( u_p - tau * gradF_p(u_p), lambda*tau );
end

figure(7)
y3 = Polar([u_p(:,1) u_p(:,2) u_p(:,3)]);
plot(t1,[y y3])

figure(8)
J = find(u_p(:,1)>1e-3);
t = (0:N-1)'/N;
s = (0:P-1)'/P;
clf; hold on;
plot(s, y, 'LineWidth', 2);
mystem(x0(I), a0(I), 'k'); % initial spikes
mystem(t(J) + Delta/(2*theta) * atan(u_p(J,3)./u_p(J,2)), u_p(J,1), 'r');  % recovered spikes
axis([0 1 0 1]);
box on;

% ADMM iteration
niter = 2000;
ADMM_x = zeros(N,3);
ADMM_z = zeros(N,3);
ADMM_u = zeros(N,3);

for i=1:niter
    [ADMM_x, ADMM_z, ADMM_u] = ProxPolarADMM(ADMM_x, ADMM_z, ADMM_u, ADMM_phi, PolarS(y), 200*tau, lambda, r, theta);
end

figure(9)
y4 = Polar([ADMM_z(:,1) ADMM_z(:,2) ADMM_z(:,3)]);
plot(t1,[y y4])

figure(10)
J = find(ADMM_z(:,1)>1e-3);
t = (0:N-1)'/N;
s = (0:P-1)'/P;
clf; hold on;
plot(s, y, 'LineWidth', 2);
mystem(x0(I), a0(I), 'k'); % initial spikes
mystem(t(J) + Delta/(2*theta) * atan(ADMM_z(J,3)./ADMM_z(J,2)), ADMM_z(J,1), 'r');  % recovered spikes
axis([0 1 0 1]);
box on;
