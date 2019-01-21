function soln = NN_model_fast(model, inputs, x)

% we can handle multiply inputs
a = inputs;

% extract W matices and compute NN
for i = 1:model.N_Ws
    
    % activation function evaluation
    z   = model.act_func(a);
    
    % the weight matrix
    W   = reshape( x( model.W_start(i):model.W_end(i) ), model.N_W_left(i), model.N_W_right(i) );
    
    % action of the weight matrix
    a   = ( W*[z; ones(1, size(inputs, 2))] ) / (model.N_W_right(i) + 1);
    
end

% last layer, outputs associated with inputs and specific x
soln.zs{1} = model.act_func(a);

end