function u = f_gianluca(N, d, dt, w)

% Initialise solution and forward map
u = zeros(N+1,1);

sdt = sqrt(dt);

for j = 1:N
    tmp = u(j)^2;
    f = d*u(j)*(1-tmp)/(1+tmp);
    u(j+1) = u(j) + dt*f + sdt*w(j);

end