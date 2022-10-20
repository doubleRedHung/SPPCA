% robust shape estimator for SPPCA (Hung, 06/12/2022)
%
% INPUT:
% X: data, n x p
% scale_a: tuning scale-parameter
% itr: max iterations
% diag_weight={0,1}, default: diag_weight = 0 for n> 3p, otherwise diag_weight =1 (use diag)  
% hard_threshold, default =0.05 
%
% OUTPUT:
% mu1: robust location estimate
% Sig1: robust shape estimate
% wt: weight for each sample point
% AR: active ratio at each "scale_a"
% d2: robus squared Mahalanobis distance

function [mu1, Sig1, wt, AR, d2] = shape_V(X, scale_a, itr, diag_weight, hard_threshold, initial)

[n,p]=size(X);

if nargin < 3
    itr=50;
end
if nargin < 4
    if n> 3*p
        diag_weight = 0;
    else
        diag_weight = 1;
    end
end   
if nargin < 5
    hard_threshold = 0.05;
end   
if nargin < 6
    mu1 = quantile(X, 0.5, 1)';
    Sig1 = diag(tauscale(X).^2);   %% diagonal initial
%     Sig1 = zeros(p);
%     for ii=1:p
%         Yi = X(:,ii);
%         Sig1(ii,ii) = tauscale(Yi)^2;
% %         Yj = X(:,ii+1:p);
% %         Sig_ipj = tauscale(repmat(Yi,1,size(Yj,2))+Yj);
% %         Sig_imj = tauscale(repmat(Yi,1,size(Yj,2))-Yj);
% %         Sig1(ii,ii+1:p) = 1/4*(Sig_ipj.^2 - Sig_imj.^2);
%     end
%     Sig1 = Sig1 + Sig1' - diag(diag(Sig1));   
else
    mu1 = initial.mu;    
    Sig1 = initial.Sigma;
end  

d=1; 
counter=1;
while d > 10^-3 && counter <= itr
    mu0 = mu1;
    Sig0 = Sig1;
    if diag_weight == 0
        l_max = svds(Sig0,1);
        wt = (X-repmat(mu0',n,1))*(Sig0+10^-8*l_max*eye(p))^-0.5;  
    else
        Sig0_inv_half = diag(diag(Sig0).^-0.5);
        wt = (X-repmat(mu0',n,1))*Sig0_inv_half;  
    end
    d2 = diag(wt*wt');
    wt = exp(-d2./scale_a);
    wt(wt <= hard_threshold) = 0;   % hard-threshold    
    
    mu1 = X'*(wt/sum(wt));
    qt = (wt.*d2)./p;
    Sig1 = (X-repmat(mu0',n,1))'*diag(wt)*(X-repmat(mu0',n,1))./sum(qt);       

    d = (norm([mu1;Sig1(:)]-[mu0;Sig0(:)])/norm([mu0;Sig0(:)]));
    counter=counter+1;
end
if counter > itr
    display(['r_0: not converge at maximum iterations ',num2str(itr)])
end

% AR = mean(wt > hard_threshold);
AR = mean(wt > 0);
end


function [sig, mu] = tauscale(X)
% Compute robust univariate location and scale estimates using the "tau
% scale" of Yohai and Zamar (1998). This is a truncated standard deviation
% of x, where x is a univariate sample to be estimated. Also return a
% weighted mean mu. Input is a data matrix X. Compute tau-scale estimates
% along the columns of X.
[n, p] = size(X);

% If X is a vector make it a column vector in all cases.
if n==1
    X = X';
    n = p;
end

c1 = 4.5;
W_c1 = @(x)((1 - (x/c1).^2).^2).*(abs(x) <= c1);
MAD = mad(X, 1);
MED = median(X, 1);
W = W_c1(bsxfun(@rdivide, bsxfun(@minus, X, MED), MAD));

% For univariate estimates with zero MAD, set weights to 1.
W(:, MAD == 0) = 1;

% Location estimate
mu = sum(X.*W)./sum(W);

% Univariate dispersion estimate
c2 = 3;
rho_c = @(x)min(x.^2, c2^2);
sig = sqrt(sum(rho_c(bsxfun(@rdivide, bsxfun(@minus, X, mu), MAD))).*(MAD.^2/n));

end

