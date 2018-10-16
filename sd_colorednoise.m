function tf = sd_colorednoise(X, Mtrain, sizeBLK)
%% Author: 
%  Alma Eguizabal, <alma.eguizabal@sst.upb.de>
%  Date: 1.8.2018 

%% Description:
%   This funtion detects the number of sources upon an array
%   in the presence of colored noise as described in [1].
%
%   Model:
%   X(t) = As(t) + n(t) , where n is a zero-mean vector with arbitraty
%                           covariance Sigma
%
%
%   Input:
%       X           -   zero-mean input data (dimensions x obervations)
%       Mtrain      -   proportion of observations used in the Train set, from 0 to 1
%       sizeBLK     -   size of the blocks in the estimated covariance
%                       matrix of the residuals (from 1 to size(X,2))
%       
%
%   Output:
%       tf          -   number of sources in X
%
%% Copyright:
% ##   Copyright (c) <2018> Signal and System Theory Group, 
% ##                        Univ. of Paderborn, http://sst.upb.de
% ##                        https://github.com/SSTGroup/Source-detection-in-colored-noise
% ##
% ##   Permission is hereby granted, free of charge, to any person
% ##   obtaining a copy of this software and associated documentation
% ##   files (the "Software"), to deal in the Software without restriction,
% ##   including without limitation the rights to use, copy, modify and
% ##   merge the Software, subject to the following conditions:
% ##
% ##   1.) The Software is used for non-commercial research and
% ##       education purposes.
% ##
% ##   2.) The above copyright notice and this permission notice shall be
% ##       included in all copies or substantial portions of the Software.
% ##
% ##   3.) Publication, distribution, sublicensing, and/or selling of
% ##       copies or parts of the Software requires special agreements
% ##       with the Signal and System Theory Group, University of Paderborn,
% ##       and is in general not permitted.
% ##
% ##   4.) Modifications or contributions to the Software must be
% ##       published under this license. 
% ##   
% ##   5.) Any publication that was created with the help of this Software  
% ##       or parts of this Software must include a citation of the paper 
% ##       referenced above.
% ##
% ##   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% ##   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
% ##   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% ##   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
% ##   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
% ##   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
% ##   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
% ##   OTHER DEALINGS IN THE SOFTWARE.
% ##
% ##   Persons using the Software are encouraged to notify the
% ##   Signal and System Theory Group at the University of Paderborn
% ##   about bugs. 
% ##
% ##
%% Function starts:

%%  1. Divide dataset in Train and Test
Nall = (1:size(X,2))';
Ntrain = round(Mtrain*size(X,2));
Nall_perm = randperm(length(Nall));
indtrain = sort(Nall_perm(1:Ntrain));
indtest = Nall(setdiff(Nall,indtrain));

Xtrain = X(:,indtrain);
Xtest = X(:,indtest);
%% 2. Build regression model
Y_data_i = Xtrain;
% 2.1 PCA anaylisis
S = (Y_data_i )*(Y_data_i)'/(size(Y_data_i,2));
[Pall,Lambda] = eig(S);
[lambdaall,indexes] = sort(real(diag(Lambda)),'descend');
Pall = Pall(:,indexes);
%% 3. Run model-order strategy in the Test residual samples
M = size(Xtest,2); % number of test samples
N = size(Xtest,1); % size of samples 
max_modes = min([N Ntrain])-1 ;
SigmaNf = zeros(max_modes,N,N)*NaN;
error_vectorF = zeros(max_modes,M,N)*NaN;
for n_modes = 1:max_modes  % For every model-order n_modes          
        Pn = Pall(:,1:n_modes);
        Ln = diag(abs(lambdaall(1:n_modes)));
        lambda_sq = sqrt(abs(diag(Ln)));
        SigmaN = eye(N); %Initial value of error noise
        diff_trace = 1;
        altr_i = 0;
        while diff_trace > 0.1 && altr_i < 10
    % Constrain alpha: alpha*sqrt(lambda)
            altr_i = altr_i + 1; 
            error_vector = zeros(M,N)*NaN;
               for id_s = 1:M             % For every sample
                    Y_i = Xtest(:,id_s);
                   % Perform regression: color noise
                    alpha = 1;
                    n_blk = sizeBLK;
                    d=round(size(SigmaN,1)/n_blk);
                    krom_mult = kron(eye(d),ones(n_blk,n_blk));
                    if size(krom_mult,1) > size(SigmaN,1)
                        krom_mult = krom_mult(1:size(SigmaN,1),1:size(SigmaN,2));
                    end
                    if size(krom_mult,1) < size(SigmaN,1)
                        d_ = size(krom_mult,1);
                        krom_mult_n = zeros(size(SigmaN));
                        krom_mult_n(1:d_,1:d_) = krom_mult;
                        krom_mult_n(d_+1:end,d_+1:end) = 1;
                        krom_mult = krom_mult_n;
                    end
                    SigmaN_ = (krom_mult.*SigmaN);
                    W  = inv(SigmaN_);
                    b_a = ((Pn'*W*Pn))\(Pn'*W)*Y_i;
                    b_a(abs(b_a)>alpha.*lambda_sq) = sign(b_a(abs(b_a)>alpha.*lambda_sq))...
                                            *alpha.*((lambda_sq(abs(b_a)>alpha.*lambda_sq)));  
                   ba = b_a;
                   error_vector(id_s,:) = (Y_i-Pn*ba);
               end % samples
        R = 1/M*(error_vector')*error_vector; 
        diff_trace = norm((R-SigmaN),'fro');
        SigmaN = R;
        %% Avoid ill-condition problems in noise matrix
        [U,S,~] = svd(SigmaN);
        diagS = diag(S);
        diagS(diagS < abs(lambdaall(end))) = abs(lambdaall(end));
        SigmaN = U*diag(diagS)*U';
        end % alternating optimization
    SigmaNf(n_modes,:,:) = SigmaN;
    error_vectorF(n_modes,:,:) = error_vector;
end %n_modes
 COST_modes = zeros(1,max_modes)*NaN;
 for mds = 1:max_modes
        dof = 2*M*mds;% penalty term, considering complex degrees of freedom
        error_vector = squeeze(error_vectorF(mds,:,:));
        SigmaNf_mds = real(krom_mult.*squeeze(SigmaNf(mds,:,:)));
        COST_modes(mds) = M*log(det(SigmaNf_mds))  + trace( (error_vector/(SigmaNf_mds))*error_vector' ) ...
                           + dof; % AIC
  end
 [~,tf] = min(real(COST_modes));

