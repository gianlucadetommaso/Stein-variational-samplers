% Performs one run of the inverse power method for sparse PCA as
% described in the paper
%
% M. Hein and T. Buehler
% An Inverse Power Method for Nonlinear Eigenproblems with Applications in 1-Spectral Clustering and Sparse PCA
% In Advances in Neural Information Processing Systems 23 (NIPS 2010)
% Available online at http://arxiv.org/abs/1012.0774
%
% Usage:
% [z,lambda,variance]= invPow(X,gamma,maxit,z)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% (C)2010-12 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de
function [z,lambda,variance]= invPow(X,gamma,maxit,z)
   
    debug=false;
    [num,dim]=size(X);
    
    Xz=X * z;
    denom=norm(Xz);
    z=z/denom;
    sigmaZ= (X'*Xz)/denom;
  
    diffLambda=inf;
    
	k=0;
       
    lambda_old=(1-gamma)*norm(z,2)+gamma*sum(abs(z));
    while (k<=maxit  && diffLambda>1E-8)
    
		k=k+1;
        
		mu=sigmaZ*lambda_old;
       
		ix = find(abs(mu)>gamma);
        z_new = zeros(dim,1);
		z_new(ix)= mu(ix)-gamma*sign(mu(ix));
        
        if(debug)
            primalobj= (1-gamma)*norm(z_new) + gamma*norm(z_new,1) - mu'*z_new;
        end
        
        Xz=X*z_new;
        denom = norm(Xz,2);
        z=z_new/denom;
        sigmaZ= (X'*Xz)/denom;
        
        %[z2,SigmaZ2,Xz2] = innerStep(X,z_new);
        %assert(norm(SigmaZ2-sigmaZ)/norm(SigmaZ2)<1E-14,sprintf('%.5g',norm(SigmaZ2-sigmaZ)/norm(SigmaZ2)));
        %assert(norm(Xz2-Xz)/norm(Xz)<1E-14,sprintf('%.5g',norm(Xz2-Xz)/norm(Xz)));
        %assert(norm(z2-z)/norm(z2)<1E-14,sprintf('%.5g',norm(z2-z)/norm(z2)));
        
        lambda=(1-gamma)*norm(z,2)+gamma*sum(abs(z));
        diffLambda=(lambda_old-lambda)/lambda_old;
        assert(diffLambda>=0 || abs(diffLambda)<1E-15);
        lambda_old=lambda;
        
    end
        
   
    variance=norm(X*z)^2/norm(z)^2;
   

end
        
        
    
   