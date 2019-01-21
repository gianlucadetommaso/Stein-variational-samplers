function [kern, g_kern] = I_kernel(x)

w = x(1:end-2,:);
n    = size(x,1);
N    = size(x,2);
g_kern = zeros(n,N,N);

if N > 1    % If more than one particle

    % Calculate squared distance between particles
    dist2 = pdist2(w',w','squaredeuclidean');

    % Calculate the squared median 
    med2 = median( dist2(dist2~=0) );

    % Set up the kernel length
    gamma = log(N)/med2;

else
    % Set up the kernel length
    gamma = 1;
end

gammawtw   = gamma * (w' * w);
dgammawtw  = diag(gammawtw);
kern   = exp( -( dgammawtw + dgammawtw' - 2*gammawtw ) );

for i = 1:N
    g_kern(:,i,:) = 2 * gamma * (x - x(:,i)) .* kern(i,:);
end
  