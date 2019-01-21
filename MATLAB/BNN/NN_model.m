function soln = NN_model(model, inputs, w)

% we can handle multiply inputs
soln.Ws = cell(model.N_Ws,   1);
soln.zs = cell(model.N_Ws+1, 1);
soln.as = cell(model.N_Ws+1, 1);
soln.df = cell(model.N_Ws+1, 1);

soln.as{1}  = inputs;

% extract W matices and compute NN
for i = 1:model.N_Ws
    
    % activation function evaluation
    soln.zs{i}  = model.act_func( soln.as{i} );
    soln.df{i}  = model.act_func_deri( soln.as{i} );
    
    % the weight matrix
    soln.Ws{i}  = reshape( w( model.W_start(i):model.W_end(i) ), model.N_W_left(i), model.N_W_right(i) );
    
    % action of the weight matrix
    soln.as{i+1}  = ( soln.Ws{i}*[soln.zs{i}; ones(1, size(inputs, 2))] ) / (model.N_W_right(i) + 1);
    
end

j = model.N_Ws + 1;
soln.zs{j}  = model.act_func( soln.as{j} );
soln.df{j}  = model.act_func_deri( soln.as{j} );

% last layer, outputs associated with inputs and specific x
soln.d  = soln.zs{j};

end