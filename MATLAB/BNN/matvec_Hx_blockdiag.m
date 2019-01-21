function Hcd = matvec_Hx_blockdiag(kern, g_kern, model, data, HI, cd)

N = length(kern);
Hpicd = zeros(model.n, N);

for k = 1:N
    Hpicd(:,k) = matvec_Fisher(model, data, HI{k}, cd);
end

Hcd = mean( Hpicd .* kern.^2 + g_kern .* (cd' * g_kern), 2); 