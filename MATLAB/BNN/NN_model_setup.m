function model = NN_model_setup(info)


% N_nodes is a vector that consists of the number of nodes in each layer
model.N_nodes   = [length(info.x_indices); info.N_int_node(:); length(info.y_indices)];

% construct indices for extracting each of W matrices
% there are (n_layer - 1) W matrices

model.N_Ws      = length(model.N_nodes)-1;
model.N_W_right = zeros(model.N_Ws, 1);
model.N_W_left  = zeros(model.N_Ws, 1);
model.N_W_els   = zeros(model.N_Ws, 1);
model.W_start   = zeros(model.N_Ws, 1);
model.W_end     = zeros(model.N_Ws, 1);

count   = 0;
for i = 1:model.N_Ws
    model.N_W_right(i)  = model.N_nodes(i) + 1;
    model.N_W_left(i)   = model.N_nodes(i+1);
    model.N_W_els(i)    = model.N_W_left(i)*model.N_W_right(i);
    model.W_start(i)    = count + 1;
    count               = count + model.N_W_els(i);
    model.W_end(i)      = count;
end

model.N_w   = count;

model.ind_log_gamma     = count + 1;
model.ind_log_lambda    = count + 2;

end