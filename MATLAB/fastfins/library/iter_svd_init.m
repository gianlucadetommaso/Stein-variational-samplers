function gsvd = iter_svd_init(U, s, max_rank)

%gsvd.iter_tol   = iter_tol;
gsvd.max_rank   = max_rank;
gsvd.tol        = max(s)*1E-15;
ind             = s >= gsvd.tol;
gsvd.s          = s(ind);
gsvd.U          = U(:,ind);
gsvd.Nsample    = 1;

end