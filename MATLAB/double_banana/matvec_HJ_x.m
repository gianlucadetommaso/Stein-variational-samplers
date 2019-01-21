function HJ_cd = matvec_HJ_x(cd, x, kern, g_kern, model, prior, obs)

N = length(kern);

HJ_cd = zeros(model.n,N);
cd  = reshape(cd, model.n, N);

for i = 1:N

    for k = 1:N
        Hpost_cd = matvec_Hpost_x(cd, x(:,k), model, prior, obs);

        sumg_kerntcd = sum( sum( reshape(g_kern(:,k,:), model.n, N) .* cd, 1), 2);
        HJ_cd(:,i) = HJ_cd(:,i) + sum(Hpost_cd .* kern(k,:) * kern(i,k), 2) ...
                            + g_kern(:,k,i) .* sumg_kerntcd;
    end
    
end

HJ_cd = HJ_cd(:)/N;
