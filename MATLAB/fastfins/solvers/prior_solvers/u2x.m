function [x, dxdu] = u2x(prior, u )
%U2X
% Transform the computation parameter u ( with prior N(m, C) ) to the 
% physical parameter x used by the PDE
%
% Tiangang Cui, 17/Jan/2014


% function [x, dxdu, dx2du2] = prior_u2x( func, u )
% For the full Hessian

if nargout == 1
    switch prior.func.type
        case{'log'}
            temp = exp(u);
            x = temp  + prior.func.log_thres;
        case{'erf'}
            x = erf(u).*prior.func.erf_scale + prior.func.erf_shift;
        otherwise
            x = u;
    end
else
    switch prior.func.type
        case{'log'}
            temp    = exp(u);
            x       = temp  + prior.func.log_thres;
            dxdu    = temp;
            %dx2du2  = temp;
        case{'erf'}
            x       = erf(u).*prior.func.erf_scale + prior.func.erf_shift;
            dxdu    = prior.func.erf_scale.*2./sqrt(pi).*exp(-u(:).^2);
            %dx2du2  = -2*dxdu.*u(:);
        otherwise
            x       = u;% + prior.shift;
            dxdu    = ones(size(x));
            %dx2du2  = zeros(size(x));
    end
end

end
