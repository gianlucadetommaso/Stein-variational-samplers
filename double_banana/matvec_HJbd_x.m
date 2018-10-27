function HJbd_cd = matvec_HJbd_x(cd, x, kern, g_kern, model, prior, obs)

N = length(kern);
Hpost_cd = zeros(model.n, N);

for k = 1:N
    Hpost_cd(:,k) = matvec_Hpost_x(cd, x(:,k), model, prior, obs);
end

HJbd_cd = mean( Hpost_cd .* kern.^2 + g_kern .* (cd' * g_kern), 2); 