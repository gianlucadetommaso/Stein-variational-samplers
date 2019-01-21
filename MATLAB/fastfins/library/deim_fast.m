function basis = deim_fast(U)
%DEIM_FAST
%
% Tiangang Cui, 17/August/2014

[n, r]          = size(U);

e               = zeros(r,1); % selection index
[~, ri]         = max(abs(U(:,1))); % extract max elements from each columns
e(1)            = ri;

%P               = spalloc(n,r,r); % selection matrix, sparse
%P(ri,1)         = 1;

final           = r;
for i = 2:r
    ind         = 1:(i-1);
    %A          = P(:,ind)'*V(:,ind);
    %tmp        = A\(P(:,ind)'*U(:,cols));
    A           = U(e(ind), ind);
    if cond(A)  > 1E15
        disp('WARNING: large condition number, need to stop the iteration');
        final   = i-1;
        break;
    end
    %[Au,As,Av]  = svd(A);
    %iA          = Au*diag(1./diag(As))*Av';
    %res         = U(:,i) - (U(:,ind)*Au) * scale_rows(Av'*U(e(ind),i), 1./diag(As));
    tmp         = A\U(e(ind),i);
    res         = U(:,i) - U(:,ind)*tmp;
    
    [~, ri]     = max(abs(res));
    e(i)        = ri;
    %P(ri,i)     = 1;
end

ind             = 1:final;
if final ~= r
    disp('adding outputs')
    basis.U         = U(:,ind);
end

[Au,As,Av]      = svd(U(e(ind), ind));
das             = diag(As);
tmp             = log10(das);
indo            = find(tmp-tmp(1) > -15);

basis.K         = Av(:,indo)*diag(1./das(indo))*Au(:,indo)';
basis.U         = U(:,ind);
basis.B         = basis.U*basis.K;
basis.e         = e(ind);
%basis.P         = P;

end