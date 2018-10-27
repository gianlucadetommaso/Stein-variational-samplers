function dw = matvec_Jty(model, HI, dz)

dw = zeros( model.N_w, 1 );

% extract W matices and compute NN

for i = model.N_Ws:-1:1
    
    da  = dz.*HI.df{i+1}; % adjoint model
    
    dW  = da * ( [HI.zs{i}', ones(size(HI.zs{i}, 2), 1)] / (model.N_W_right(i) + 1) );
    dw(model.W_start(i):model.W_end(i)) = dW(:);
    
    dz = ( HI.Ws{i}(:, 1:end-1)'*da ) / (model.N_W_right(i) + 1);
    
end

end

