function r = two_loops(Y, S, num, q, type)
% TWO_LOOPS    two loops revursion for the BFGS update, using prior as
%              initial guess recent iterations are in front of Y and S
%
% Tiangang Cui, 16/Mar/2013

%r = q;
%return

N = size(Y,2);
if num < N
    N = num;
end

rho = zeros(N,1);
alpha = zeros(N,1);
for i = 1:N
    rho(i) = 1./(Y(:,i)'*S(:,i));
end

%if N > 1
%    rho(end)
%end

switch type
    case {'H'}
        for i = 1:N
            alpha(i) = rho(i)*(Y(:,i)'*q);
            q = q - alpha(i)*S(:,i);
        end
        r = q;
        for i = N:-1:1
            beta = rho(i)*(S(:,i)'*r);
            r = r + (alpha(i) - beta)*Y(:,i);
        end
    case {'iH'}
        for i = 1:N
            alpha(i) = rho(i)*(S(:,i)'*q);
            q = q - alpha(i)*Y(:,i);
        end
        r = q;
        for i = N:-1:1
            beta = rho(i)*(Y(:,i)'*r);
            r = r + (alpha(i) - beta)*S(:,i);
        end
end