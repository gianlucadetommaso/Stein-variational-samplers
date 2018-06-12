function basis = deim(U)
%DEIM
%
% Tiangang Cui, 17/August/2014

[n, r]          = size(U);
cols_out        = 1:r; % for the residual 
cols            = 1:r;  % 

e               = zeros(r,1); % row selection index

[vals, ris]     = max(abs(U)); % extract max elements from each columns
[~, ci]         = max(vals);
ri              = ris(ci);
cols_out        = cols_out(cols_out~=ci);
cols_in         = setdiff(cols, cols_out);
e(ci)           = ri;

for i = 2:r
    A           = U(e(cols_in), cols_in);
    if cond(A)  > 1E10
        disp('WARNING: large condition number, need to stop the iteration');
        break;
    end
    tmp         = A\U(e(cols_in), cols_out);
    R           = U(:,cols_out) - U(:,cols_in)*tmp;
    
    [vals, ris] = max(abs(R));
    [~, j]      = max(vals);
    ci          = cols_out(j);
    ri          = ris(j);
    cols_out    = cols_out(cols_out~=ci);
    cols_in     = setdiff(cols, cols_out);
    e(ci)       = ri;
end


k               = length(cols_in);
if k ~= r
    disp('adding outputs')
    basis.U     = U(:, cols_in);
end

basis.V         = U(:, cols_in);
basis.K         = inv(U(e(cols_in), cols_in));
basis.B         = basis.V*basis.K;
basis.e         = e(cols_in);

%basis.P        = sparse(basis.e, (1:k)', ones(k,1), n, r);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{

function basis = deim_2(U)
%DEIM
%
% Tiangang Cui, 17/August/2014

[n, r]          = size(U);
cols            = 1:r; % a selection vector

V               = zeros(n,r); % output basis
e               = zeros(r,1); % selection index
p               = zeros(r,1); % selection index

[vals, ris]     = max(abs(U)); % extract max elements from each columns
[~, ci]         = max(vals);
ri              = ris(ci);
V(:,1)          = U(:,ci);
cols            = cols(cols~=ci);
e(1)            = ri;
p(1)            = ci;


%[~, ri]        = max(abs(V(:,1)));
P              = spalloc(n,r,r); % selection matrix, sparse
P(ri,1)        = 1;

final           = r;
for i = 2:r
    ind         = 1:(i-1);
    %A          = P(:,ind)'*V(:,ind);
    %tmp        = A\(P(:,ind)'*U(:,cols));
    A           = V(e(ind), ind);
    if cond(A)  > 1E10
        disp('WARNING: large condition number, need to stop the iteration');
        final   = i-1;
        break;
    end
    tmp         = A\U(e(ind),cols);
    R           = U(:,cols) - V(:,ind)*tmp;
    
    [vals, ris] = max(abs(R));
    [val, j]      = max(vals);
    ci          = cols(j);
    ri          = ris(j);
    V(:,i)      = U(:,ci);
    cols        = cols(cols~=ci);
    e(i)        = ri;
    p(i)        = ci;
    %[~, ri]    = max(abs(R(:,j)))
    P(ri,i)    = 1;
    val
end

ind             = 1:final;
basis.K         = inv(V(e(ind),ind));
basis.V         = V(:,ind);
basis.B         = basis.V*basis.K;
basis.e         = e(ind);
basis.p         = p(ind);

basis.P        = P;

end

%}