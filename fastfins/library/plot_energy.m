function ind = plot_energy( d, tol )
%PLOT_ENERGY
%
% Tiangang Cui, 1/June/2014

cd  = cumsum(d)/sum(d);
ind = sum( cd < 1-tol );

%{
j   = d>1e-20;


figure
subplot(1,2,1)
semilogy(d(j), 'b');hold on
semilogy([ind ind], [min(d(j)), max(d(j))], 'k')
ylabel('eigenvalue')
xlabel('index')
axis tight
set(gca, 'xlim', [1, length(j)]);
subplot(1,2,2)
plot(cd(j), 'r');hold on
%plot([1 length(d)], 1 - [tol, tol], 'k')
plot([ind ind], [0, 1], 'k')
set(gca, 'ylim', [0, 1]);
set(gca, 'xlim', [1, length(j)]);
ylabel('sum of eigenvalues')
xlabel('index')
%}

end

