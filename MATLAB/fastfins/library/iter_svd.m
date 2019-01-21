function gsvd = iter_svd(gsvd, U, s)

 T              = gsvd.U'*U;
[Q, R]          = qr(U - gsvd.U*T, 0);
 tmp1           = scale_cols([T; R], s);
 tmp2           = tmp1*tmp1' + diag([gsvd.s(:).^2*gsvd.Nsample; zeros(length(s), 1)]);
[Phi, D]        = eig(tmp2);
[s_glo, ind]    = sort(sqrt(real(diag(D)))/sqrt(gsvd.Nsample+1), 'descend');
 U_glo          = [gsvd.U, Q]*Phi(:, ind);
%{
% full update
 Vmat           = [scale_cols(gsvd.U, gsvd.s)*sqrt(gsvd.Nsample), scale_cols(U, s)];
[U_glo, S_glo]  = svd(Vmat/sqrt(gsvd.Nsample+1),'econ'); % global SVD
 s_glo          = diag(S_glo); 
%}

ind             = s_glo>=gsvd.tol;
gsvd.Nsample    = gsvd.Nsample + 1;

if length(ind) > gsvd.max_rank
    gsvd.U      = U_glo(:,ind(1:1000));
    gsvd.s      = s_glo(ind(1:1000));
else
    gsvd.U      = U_glo(:,ind);
    gsvd.s      = s_glo(ind);
    
end

%disp(length(gsvd.s))

end