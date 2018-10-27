function RMSE = validate(model, data, estimate, plot_flag)

% re-producing traning data
tmp = NN_model_fast(model, data.xs, estimate);
ys  = tmp.zs{end} * data.std(data.y_indices) + data.min(data.y_indices);
tys = data.ys * data.std(data.y_indices) + data.min(data.y_indices);

if plot_flag >= 2
    figure
    subplot(2,2,1)
    plot(tys , '.')
    title('training data')
    subplot(2,2,2)
    plot(ys, '.')
    title('model')
    subplot(2,2,3)
    plot(tys, '.')
    hold on
    plot(ys, '.')
    title('blue: data, red: model')
    subplot(2,2,4)
    plot(tys, ys, '.')
    xlabel('data')
    ylabel('model')
end

% cross validation
tmp = NN_model_fast(model, data.validate_xs, estimate);
pre = tmp.zs{end} * data.std(data.y_indices) + data.min(data.y_indices);
vys = data.validate_ys * data.std(data.y_indices) + data.min(data.y_indices);
RMSE = sqrt(mean((pre - vys).^2));

if plot_flag >= 1
    figure
    subplot(2,2,1)
    plot(vys, '.')
    title('validation')
    subplot(2,2,2)
    plot(pre, '.')
    title('model')
    subplot(2,2,3)
    plot(vys, '.')
    hold on
    plot(pre, '.')
    title('blue: data, red: model')
    subplot(2,2,4)
    plot(vys, pre, '.')
    xlabel('data')
    ylabel('model')
end

end