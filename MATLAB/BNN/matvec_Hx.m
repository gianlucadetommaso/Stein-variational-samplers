function Hcd = matvec_Hx(kern, g_kern, model, data, HI, cd)

N = length(kern);

Hcd = zeros(model.n,N);
cd  = reshape(cd, model.n, N);

for i = 1:N

    for k = 1:N
        Hpicd = matvec_Fisher(model, data, HI{k}, cd);

        sumg_kerntcd = sum( sum( reshape(g_kern(:,k,:), model.n, N) .* cd, 1), 2);

        Hcd(:,i) = Hcd(:,i) + sum(Hpicd .* kern(k,:) * kern(i,k), 2) ...
                            + g_kern(:,k,i).* sumg_kerntcd;
    end
    
end

Hcd = Hcd(:)/N;
