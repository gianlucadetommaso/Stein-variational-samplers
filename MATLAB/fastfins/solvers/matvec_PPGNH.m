function w = matvec_PPGNH(model, obs, prior, HI, dv)
%MATVEC_PPH
%
% compute the matvec with PPH
%
% Tiangang Cui, 19/Mar/2014

du      = matvec_prior_L(prior, dv); % transform to u

temp    = repmat(HI.dxdu,1,size(dv,2));
dx      = temp.*du; % transform to physical space x

wx      = matvec_GNH(model, obs, HI, dx); % matvec with GNH for the model
wu      = wx.*temp; % transform the matvec to u

w       = matvec_prior_Lt(prior, wu); % transform back to v

end

function wx = matvec_GNH(model, obs, HI, dx)

n       = size(dx, 2);
wx      = zeros(size(dx));

for i = 1:n
    tmp     = matvec_Jx (model, HI, dx(:,i))./(obs.std.^2);
    wx(:,i) = matvec_Jty(model, HI, tmp);
end

end