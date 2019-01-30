function f2=TV_min(a1,lambda,iter)
% minimization of |F(K*x) + G(x)|
% Regularization parameter.
if nargin<2
lambda = 100;iter=300;
elseif nargin<3
    iter=300;
end

K = @(x)grad(x); %define gradient norm(K(x),1) is thus the TV constraint
KS = @(x)-div(x);

Amplitude = @(u)sqrt(sum(u.^2,3));
F = @(u)lambda*sum(sum(Amplitude(u)));
%G = @(x)1/2*norm(sinogram-myRadon(x,thetas),'fro')^2;
G = @(x)1/2*norm(a1-x,'fro')^2;

% The proximity operator of |F| is the vectorial soft thresholding.
Normalize = @(u)u./repmat( max(Amplitude(u),1e-10), [1 1 2] );
ProxF = @(u,tau)repmat( perform_soft_thresholding(Amplitude(u),lambda*tau), [1 1 2]).*Normalize(u);
ProxFS = compute_dual_prox(ProxF);

% The proximity operator of G.
ProxG = @(x,tau)(x+tau*a1)/(1+tau);

options.report = @(x)G(x) + F(K(x));

% Run the ADMM algorihtm.

options.niter = iter;
f2 = perform_admm(a1, K,  KS, ProxFS, ProxG, options);


% Display image.

% figure (3)
% subplot(121)
% imageplot(a1)
% subplot(122)
% imageplot(f2);

clear F G ProxG ProxFS K KS Amplitude lambda Normalize options ProxF 