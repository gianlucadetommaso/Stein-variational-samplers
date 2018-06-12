% Computes the sparse PCA components using the nonlinear inverse power
% method described in the paper
%
% M. Hein and T. Buehler
% An Inverse Power Method for Nonlinear Eigenproblems with Applications 
% in 1-Spectral Clustering and Sparse PCA
% In Advances in Neural Information Processing Systems 23 (NIPS 2010)
% Available online at http://arxiv.org/abs/1012.0774
%
% Usage:
% [cards,vars,Z]= sparsePCA(X,card);
% [cards,vars,Z]= sparsePCA(X,card_min,card_max);
% [cards,vars,Z]= sparsePCA(X,card_min,card_max,numRuns);
% [cards,vars,Z]= sparsePCA(X,card_min,card_max,numRuns,verbosity);
%
% X : data matrix (num x dim)
% card : desired number of non-sparse components of output (cardinality)
% card_min,card_max : computes all vectors with cardinality values in 
%     intervall [card_min,card_max] (default: card_min=card_max)
% numRuns : number of runs of inverse power method with random 
%     initialization (default: 10)
% verbosity [0-2]: determines how much information is displayed (default: 1)
%
% cards : the cardinalities (number of nonzero components) of the 
%	returned vectors 
% vars : the corresponding vectors
% Z : the sparse principal components
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% (C)2010-12 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de
function [all_cards,all_vars,all_Zs]= sparsePCA(X,card_min,card_max,numRuns,verbosity)

    if(nargin<5)
        verbosity=1;
    end    
    if nargin<4
		numRuns=10;
	end
	if nargin<3
		card_max=card_min;
    end
    
    assert(card_min>0,'Wrong usage. Cardinality has to positive.');
    assert(card_max>=card_min,'Wrong usage. card_max cannot be smaller than card_min.');
    assert(numRuns>=0,'Wrong usage. numRuns cannot be negative.');
    assert(card_max<=size(X,2),'Wrong usage. Cardinality can not be larger than dim.');
	
    [num,dim]=size(X);
	gam_left=0;
	gam_right=1;
    
    % center input and compute startvector
    X=X-repmat(mean(X,1),num,1);
    norm_a_i=zeros(dim,1);
    for i=1:dim,
        norm_a_i(i)=norm(X(:,i));
    end
    [rho_max,i_max]=max(norm_a_i);
    start1=zeros(dim,1);
    start1(i_max)=1;
    
    
    % output
    all_cards=zeros(card_max-card_min+1,1);
	all_gammas=zeros(card_max-card_min+1,1);
	all_vars=zeros(card_max-card_min+1,1);
	all_Zs=zeros(dim,card_max-card_min+1);
    all_lambdas=zeros(card_max-card_min+1,1);
    all_found=zeros(card_max-card_min+1,1);
	
	
    card0=round((card_max+card_min)/2);
	
    % searches for a vector with cardinality card0 via binary search (and
    % stores the solutions for other cardinalities it finds along the way)
	[all_cards,all_vars,all_Zs,all_gammas,all_lambdas,all_found]= binSearch(X,card0,gam_left,gam_right,numRuns,start1,all_cards,all_vars,all_Zs,all_gammas,all_lambdas,all_found,card_min,card_max,verbosity);
	
    % repeat this until all gaps are filled
	ix1=find(all_found==0,1);
	while(~isempty(ix1))
		
		card_min_temp=card_min-1+ix1;
		if(card_min_temp>card_min)
			gam_right=all_gammas(card_min_temp-card_min);
		else 
			gam_right= 1;
		end
		
		ix2=find(all_found(ix1:end)>0,1);
        if(~isempty(ix2))
            if all_found(ix2)<inf
                card_max_temp=card_min-1+ix1-1+ix2-1;
                gam_left=all_gammas(card_max_temp-card_min+2);
            else
                card_max_temp=card_min-1+ix1-1+ix2-1;
                gam_left=0;
            end
        else
            card_max_temp=card_max;
			gam_left=0;
        end
        
        
		card0=round((card_max_temp+card_min_temp)/2);
	
        if (verbosity>1)
            fprintf('card_min_temp= %d card_max_temp= %d gam_left=%.5g gam_right=%.5g\n',card_min_temp,card_max_temp,gam_left,gam_right);
        end
        
        [all_cards,all_vars,all_Zs,all_gammas,all_lambdas,all_found]= binSearch(X,card0,gam_left,gam_right,numRuns,start1,all_cards,all_vars,all_Zs,all_gammas,all_lambdas,all_found,card_min,card_max,verbosity);
        
        ix1=find(all_found==0,1);
    end
    
   
    noChange=false;
    all_vars_old=all_vars;
    while(~noChange)
    
        % check if variance is monotonically increasing
        curvar=all_vars(1);
        curz=all_Zs(:,1);
        for k=2:length(all_vars)
            newvar=all_vars(k);
            if(newvar<curvar || all_cards(k)==inf)
                % take sparsity pattern from previous one + add one nonzero
                % component -> increasse in variance guaranteed
                norm_a_i=zeros(dim,1);
                pattern =  abs(curz)>0;
                ind=find(pattern==0);
                temp=X*curz;temp=temp/norm(temp);
                for i=1:length(ind),
                    norm_a_i(ind(i))=(X(:,ind(i))'*temp)^2;
                end
                [rho_max,i_max]=max(norm_a_i);
                pattern(i_max)=1;
                z=optVar(X,pattern);
                newvar2=norm(X*z)^2/norm(z)^2;
     
                all_vars(k)=newvar2;
                all_Zs(:,k)=z;
                all_cards(k)=sum(pattern);
               
                % check if we find something better via invpow
                card0=card_min-1+k;
                gam_left=0;
                gam_right=1;
                [all_cards,all_vars,all_Zs,all_gammas,all_lambdas,all_found]= binSearch(X,card0,gam_left,gam_right,numRuns,start1,all_cards,all_vars,all_Zs,all_gammas,all_lambdas,all_found,card_min,card_max,verbosity);
         
            end
            
            curvar=all_vars(k);
            curz=all_Zs(:,k);
           
        end
    
        if  norm(all_vars-all_vars_old,inf)/norm(all_vars(all_vars<inf),inf)< 1E-15
            noChange=true;
        end
        all_vars_old=all_vars;
        if verbosity>1
            fprintf('NoChange=%d  normdiff=%.15g\n',noChange,norm(all_vars-all_vars_old,inf));
        end
    end
    
              
   
    
    if verbosity>1
        
        noDecr=true;
        curvar=all_vars(1);
        for k=2:length(all_vars)
            newvar=all_vars(k);
            if newvar< curvar 
                noDecr=false;
            end
            curvar=newvar;
        end
        fprintf('NoChange=%d NoDecrease=%d \n',noChange,noDecr);
    end
	
end


% searches for a vector with cardinality card0 via binary search (and
% stores the solutions for other cardinalities it finds along the way)
function [all_cards,all_vars,all_Zs,all_gammas,all_lambdas,all_found]= binSearch(X,card0,gam_left,gam_right,numRuns,start1,all_cards,all_vars,all_Zs,all_gammas,all_lambdas,all_found,card_min_global,card_max_global,verbosity)


    [num,dim]=size(X);

    maxit=100;

    epsilon=1E-6;
    found=false;
    
    splitpoint=0.5;
    gam=splitpoint*gam_left+(1-splitpoint)*gam_right;
    

    while (~found && gam_right-gam_left>epsilon)
        
        Z=zeros(length(start1),numRuns+1);
        variance=zeros(numRuns+1,1);
        card=zeros(numRuns+1,1);
        lambda=zeros(numRuns+1,1);
        
        % perform sparse PCA with current value of gamma
        [z,lambda(1)]=invPow(X,gam,maxit,start1);
        z=optVar(X,abs(z)>0);
        Z(:,1)=z;
        variance(1)=norm(X*z)^2/norm(z)^2;
        card(1)=sum(abs(z)>0);
        
        for l=1:numRuns
            start=randn(dim,1);
            [z,lambda(l+1)] = invPow(X,gam,maxit,start);
            z=optVar(X,abs(z)>0);
            Z(:,l+1)=z;
            variance(l+1)=norm(X*z)^2/norm(z)^2;
            card(l+1)=sum(abs(z)>0);
        end
      
        
        
       
        
        
        %find best one for each cardinality
         card_unique=unique(card);
         variance_unique=zeros(length(card_unique),1);
         votes=zeros(length(card_unique),1);
         Z_unique=zeros(length(start),length(card_unique));
         lambda_unique=zeros(length(start),1);
         
         for l=1:length(card_unique)
             ind_temp=find(card==card_unique(l));
             variance_temp=variance(ind_temp);
             ind=find(max(variance_temp)==variance_temp,1);
             variance_unique(l)=variance_temp(ind);
             lambda_temp=lambda(ind_temp);
             lambda_unique(l)=lambda_temp(ind);
             votes(l)=length(ind_temp);
             Z_unique(:,l)=Z(:,ind_temp(ind));
         end
         
         
         %consider the cardinality corresponding to smallest lambda
         ind=find(lambda==min(lambda),1);
         bestcard=card(ind);
           
         if card0==bestcard
            found=true;
         else
             % update gamma boundaries
             if (bestcard>card0)
                 gam_left=gam;
             elseif (bestcard<card0)
                 gam_right=gam;
             end
         end
        
         % discard all values which are outside the cardinality range
         ind= find(card_unique>=card_min_global & card_unique<=card_max_global);
        
        if(verbosity>1)
            ind2=find(card_unique<card_min_global | card_unique>card_max_global);
            if(~isempty(ind2))
                for l=1:length(ind2)
                    fprintf('Skipping solution with cardinality %d\n',card_unique(ind2(l)));
                end
            end
        end
              
        card_unique=card_unique(ind);
        variance_unique=variance_unique(ind);
        Z_unique=Z_unique(:,ind);
        lambda_unique=lambda_unique(ind);
               
        % store best results
        for l=1:length(card_unique)
            curcard=card_unique(l);
            curvar=variance_unique(l);
            
            if curcard==bestcard
                 all_found(bestcard-card_min_global+1)=bestcard;
            end
            
            if all_vars(curcard-card_min_global+1)<curvar
                
                if verbosity>0
                    if (all_cards(curcard-card_min_global+1)==0)
                        fprintf('Found solution with cardinality %d\n',curcard);
                    else
                        fprintf('Improved solution with cardinality %d\n',curcard);
                    end
                end
                all_cards(curcard-card_min_global+1)=curcard;
				all_vars(curcard-card_min_global+1)=curvar;
				all_Zs(:,curcard-card_min_global+1)=Z_unique(:,l);
				all_gammas(curcard-card_min_global+1,:)=gam;
                all_lambdas(curcard-card_min_global+1)=lambda_unique(l);
             
                
            end
        end
       
     
        gam=splitpoint*gam_left+(1-splitpoint)*gam_right;
    
	
	if(verbosity>1)
        fprintf('gam_left= %.3g gam=%.3g gam_right= %.3g mincard=%d maxcard=%d bestcard=%d card0=%d\n', ...
            gam_left,gam,gam_right,min(card_unique), max(card_unique), bestcard,card0);
    end
        
    end
    
    % if no vector with cardinality card0 could be found, set entry to inf
    if(~found)
        % (might have been found as suboptimal solution)
        if all_cards(card0-card_min_global+1)==0
            all_cards(card0-card_min_global+1)=Inf;
            all_vars(card0-card_min_global+1)=Inf;
        end
        if verbosity>0
            fprintf('Skipping solution with cardinality %d\n',card0);
        end
        all_found(card0-card_min_global+1)=inf;
    end
end	



% finds the optimal variance for a given sparsity pattern ind	
function z=optVar(X,ind)

    if (sum(ind)==1)
        z=double(ind);
    else
        X2=X(:,ind);
        Sigma=X2'*X2;
    
        opts.disp=0;
        [z_temp,var1]=eigs(Sigma,1,'LM',opts);
    
        z=zeros(size(X,2),1);
        z(ind)=z_temp;
    end
    
    
end
    	
	






