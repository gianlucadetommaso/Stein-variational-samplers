function [kern, g_kern] = kernel_full(x, H)

n    = size(x,1);
N    = size(x,2);
g_kern = zeros(n,N,N);

Hx     = H * x;
xtHx   = x' * Hx;
dxtHx  = diag(xtHx);
kern   = exp( -0.5 * ( dxtHx + dxtHx' - 2*xtHx ) );

for i = 1:N
    g_kern(:,i,:) = (Hx - Hx(:,i)) .* kern(i,:);
end
  