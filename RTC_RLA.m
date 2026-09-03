addpath(genpath('Noise'));
addpath(genpath('data'));
addpath(genpath('tool'));

seed = 49;
rng(seed, 'twister');
fprintf('Random seed has been set to: %d\n', seed);
dataRoad = ['guangzhou'];
load(dataRoad);
X0=permute(GuangZhou,[2 3 1]);
%X0=X;
dim = size(X0);
p = 11;
missingrate = 0.5;
missingway = 'Random';
switch missingway
case 'Random'
Pomega = round(rand(dim(1),dim(2),dim(3)) + 0.5 - missingrate);
case 'Non-random2'
A = round(rand(dim(1), dim(3)) + 0.5 - missingrate);
B = kron(A, ones(1,dim(2)));
Pomega = reshape(B,[dim(1),dim(2),dim(3)]);
end

%% LG noise settings
L_beta = 0;
G_std  = 3;
noise  = noise_tensor(dim, L_beta, G_std);

scenario = 1;          % 1: severe high-frequency corruption
                       % 2: catastrophic low-frequency corruption

%anomal=hybrid_collapse_anomaly_nospan(X0, scenario, seed);
anomal = event_disruption_anomaly(X0, scenario, seed);
% Keep anomalies only at observed entries
anomal = Pomega .* anomal;
Y = X0 + noise +anomal;

PY = Pomega .* Y;

lambda1 = 1/sqrt(max(dim(3),dim(2))*dim(1));
lambda2 = 1/sqrt(max(dim(3),dim(2))*dim(1)) ;


Pomegac  = 1 - Pomega;
max_iter = 200;
rho      = 1.1;
mu1      = 1e-6;
mu2      = 1e-6;
max_mu   = 1e10;
detail   = 1;
eps      = 1e-6;

%% T1: along mode 1; T2: along mode 2
%% T1: only along mode 1
T1_hat       = fft([1, -1, zeros(1, dim(1)-2)]);
T1_hat_sq    = abs(T1_hat).^2;
T1_hat_sq_3d = reshape(T1_hat_sq, [dim(1), 1, 1]);
%% FFT setting for AR operator D
v   = zeros(dim(3), dim(1));
D   = zeros(dim(1), dim(1), dim(3));
TAR = zeros(dim);
vv  = repmat(num2vetT(1, p+1), [dim(3), 1]);

outer_tol      = 1e-1;
outer_max_iter = 20;
tol            = 1e-2;

%% Variable initialization
X = PY;
%min_outer_before_stop = 4;
outer_iter      = 0;
outer_converged = false;
E1_prev = zeros(dim);
E2_prev = zeros(dim);
while ~outer_converged && outer_iter < outer_max_iter
outer_iter = outer_iter + 1;
if detail, fprintf('\n=== Outer iteration %d ===\n', outer_iter);end

X_prev = X;

%% Reset ADMM variables (without Z and Upsilon)
mu1_current = mu1;
mu2_current = mu2;
G      = zeros(dim);
E1     = zeros(dim);
E2     = zeros(dim);
K      = zeros(dim);
Lambda = zeros(dim);
Gamma  = zeros(dim);

%% Update AR parameters
MMM = reshape(X, [dim(1)*dim(2), dim(3)]);
vv  = fliplr(arburg(MMM, p));
for i = 1:dim(3)
    v(i,:)     = [vv(i,:), zeros(1, dim(1)-p-1)];
    D(:,:,i)   = circvet2mat(v(i,:));
    TAR(:,:,i) = repmat(abs(fft(v(i,:)')).^2, [1, dim(2)]);
end

MAE_start = (1/prod(dim)) * sum(abs(X0 - X), 'all');
if detail, fprintf('  MAE before inner ADMM: %.6f\n', MAE_start); end

iter = 0;
inner_converged = false;

while ~inner_converged && iter < max_iter
    iter = iter + 1;
    Xk  = X;
    E1k = E1;
    E2k = E2;

    %% (1) Update X
    for i = 1:dim(3)
        W(:,:,i) = D(:,:,i)' * (G(:,:,i) - Gamma(:,:,i)/mu2_current) + ...
                   PY(:,:,i) - E1(:,:,i) - E2(:,:,i) - K(:,:,i) + Lambda(:,:,i)/mu1_current;
        X(:,:,i) = real(ifft(fft(W(:,:,i)) ./ (1 + TAR(:,:,i))));
    end

    %% Compute DX
    DX = zeros(dim);
    for i = 1:dim(3)
        DX(:,:,i) = D(:,:,i) * X(:,:,i);
    end

    %% (2) Update G (TrNN)
    J    = permute(DX + Gamma/mu2_current, [3 1 2]);
    [JJ] = prox_TTNN(J, 1/mu2_current, 5);
    G    = permute(JJ, [2 3 1]);

    %% (3) Update E1 (l1, absorbs Laplace noise)
    E1 = prox_l1(PY - X - E2 - K + Lambda/mu1_current, lambda1/mu1_current);

    %% (4) Update E2 (l2, direct closed-form solution, no Z required)

% min mu1/2||E2-RE2||² + lambda2/2*||E2×₁T1||²
% Apply FFT to E2 only along dim1; the denominator is:
% mu1 + lambda2*|T1̂|²
RE2   = PY - X - E1 - K + Lambda/mu1_current;
denom = mu1_current + lambda2 .* T1_hat_sq_3d;
E2 = real(ifft( ...
mu1_current .* fft(RE2, [], 1) ./ denom, ...
[], 1));

    %% (5) Update K (projection onto Ω^c)
    K = PY - X - E1 - E2 + Lambda/mu1_current;
    K = Pomegac .* K;

    %% Evaluation metrics
    MAE  = (1/prod(dim)) * sum(abs(X0 - X), 'all');
    RMSE = sqrt((1/prod(dim)) * sum((X0 - X).^2, 'all'));
    

    %% Stopping criterion
    dY   = PY - X - E1 - E2 - K;
    chgX = max(abs(Xk(:)  - X(:)));
    chgE1= max(abs(E1k(:) - E1(:)));
    chgE2= max(abs(E2k(:) - E2(:)));
    chg  = max([chgX, chgE1, chgE2, max(abs(dY(:)))]);
    if chg < tol, inner_converged = true; end

    if detail && (iter==1 || mod(iter,10)==0)
        err = norm(dY(:),'fro');
        disp(['  Inner iteration ' num2str(iter) ', mu1=' num2str(mu1_current) ...
              ', mu2=' num2str(mu2_current) ', chgX=' num2str(chgX) ...
              ', err=' num2str(err) ...
              ', MAE=' num2str(MAE) ', RMSE=' num2str(RMSE)]);
    end

    %% (6) Update multipliers (without Upsilon)
    Lambda      = Lambda + mu1_current * dY;
    Gamma       = Gamma  + mu2_current * (DX - G);
    mu1_current = min(rho*mu1_current, max_mu);
    mu2_current = min(rho*mu2_current, max_mu);
end

MAE_end = (1/prod(dim)) * sum(abs(X0 - X), 'all');
if detail
    fprintf('  Inner ADMM finished: %d iterations, MAE improvement: %.6f -> %.6f\n', iter, MAE_start, MAE_end);
end

% Save the residuals identified in this round for updating D in the next round
E1_prev = E1;
E2_prev = E2;
rel_change = norm(X(:)-X_prev(:)) / (norm(X_prev(:))+eps);
if detail, fprintf('  Relative change of X: %.4e\n', rel_change); end
if rel_change < outer_tol
outer_converged = true;
if detail, fprintf('Outer loop converged!\n'); end
end
end

fprintf('\nFinal MAE:  %.6f\n', MAE); 
fprintf('Final RMSE: %.6f\n', RMSE);
