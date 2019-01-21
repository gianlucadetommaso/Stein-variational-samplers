function dz = matvec_Jx(model, HI, dw)

% the perturbation is zero in the first layer
dz  = zeros(size(HI.zs{1}));

% extract W matices and compute linearised forward model
for i = 1:model.N_Ws
    
    % the weight matrix
    dW  = reshape( dw( model.W_start(i):model.W_end(i) ), model.N_W_left(i), model.N_W_right(i) );
    
    % action of the weight matrix
    % soln.as{i+1}  = ( soln.Ws{i}*[soln.zs{i}; 1] ) / (model.N_W_right(i) + 1);
    
    % local perturbation of a
    da = ( dW*[HI.zs{i}; ones(1, size(HI.zs{i}, 2))] + HI.Ws{i}(:, 1:end-1)*dz ) / (model.N_W_right(i) + 1);
    
    % local perturbation activation function evaluation
    % soln.zs{i}  = model.act_func( soln.as{i} );
    % soln.df{i}  = model.act_func_deri( soln.as{i} );
    
    dz  = HI.df{i+1}.*da;
end

end
