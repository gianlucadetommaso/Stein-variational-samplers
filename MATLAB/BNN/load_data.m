function data = load_data(info)
% load data file
tmp         = load(info.data_base, info.data_name);
raw_data    = tmp.(info.data_name);

data.data_size  = info.data_size;
data.t_ratio    = info.t_ratio;
data.x_indices  = info.x_indices;
data.y_indices  = info.y_indices;

% when ReLU function is used, set the minimum to zero
if info.normal_on
    data.min    = min(raw_data);
    data.max    = max(raw_data);
    data.mean   = mean(raw_data);
    data.std    = std(raw_data);
    raw_data    = ( raw_data - repmat(data.min, size(raw_data, 1), 1) ) ./ repmat(data.std, size(raw_data, 1), 1);
end

% index for training and validation
ind             = randperm(data.data_size, data.data_size);
t_size          = ceil(info.data_size*data.t_ratio);
data.t_indices  = ind(1:t_size);
data.v_indices  = ind((t_size+1):end);

data.xs     = raw_data(data.t_indices, data.x_indices)';
data.ys     = raw_data(data.t_indices, data.y_indices)';
data.N_y    = length(data.y_indices) * length(data.t_indices);

data.validate_xs   = raw_data(data.v_indices, data.x_indices)';
data.validate_ys   = raw_data(data.v_indices, data.y_indices)';

end