function dw = matvec_Jty_safe(model, HI, dz)

dw = zeros( model.N_w, size(dz, 2) );

% extract W matices and compute NN

for i = model.N_Ws:-1:1
    
    da  = dz.*HI.df{i+1}; % adjoint model
    
    for j = 1:size(dz, 2)
        dW  = da(:,j) * ( [HI.zs{i}(:,j)', 1] / (model.N_W_right(i) + 1) );
        dw(model.W_start(i):model.W_end(i), j) = dW(:);
    end
    
    dz = ( HI.Ws{i}(:, 1:end-1)'*da ) / (model.N_W_right(i) + 1);
    
end

dw  = sum(dw, 2);

end



%{

% the perturbation is zero in the first layer
dz  = zeros(size(model.zs{1}));

% extract W matices and compute linearised forward model
for i = 1:model.N_Ws
    
    % the weight matrix
    dW  = reshape( dx( model.W_start(i):model.W_end(i) ), model.N_W_left, model.N_W_right );
    
    % action of the weight matrix
    % soln.as{i+1}  = ( soln.Ws{i}*[soln.zs{i}; 1] ) / (model.N_W_right(i) + 1);
    
    % local perturbation of a
    da = ( dW*[model.zs{i}; 1] + HI.Ws{i}(:, 1:end-1)*dz ) / (model.N_W_right(i) + 1);
    
    % local perturbation activation function evaluation
    % soln.zs{i}  = model.act_func( soln.as{i} );
    % soln.df{i}  = model.act_func_deri( soln.as{i} );
    
    dz  = soln.df{i+1}.*da;
end

%}