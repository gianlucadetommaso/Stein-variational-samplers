function [kern, g_kern] = H_kernel(model, data, HI, x)

n    = size(x,1);
N    = size(x,2);
g_kern = zeros(n,N,N);

Hx = zeros(model.n, N);
for k = 1:N
    Hx = Hx + matvec_Fisher(model, data, HI{k}, x);
end
Hx = Hx / N;

xtHx   = x' * Hx;
dxtHx  = diag(xtHx);
kern   = exp( -0.5 * ( dxtHx + dxtHx' - 2*xtHx ) );

for i = 1:N
    g_kern(:,i,:) = (Hx - Hx(:,i)) .* kern(i,:);
end
  