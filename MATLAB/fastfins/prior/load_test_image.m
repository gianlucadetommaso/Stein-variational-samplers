function true_x = load_test_image(prior, test_image, output)
%INIT_TEST_IMAGE  
%
% initializes the test image for distributed parameters
%
% Tiangang Cui, 11/May/2014

if ~isfield(test_image, 'type')
    test_image.type     = 'Prior';
end
if ~isfield(test_image, 'base')
    test_image.base     = 2;
end
if ~isfield(test_image, 'range')
    test_image.range    = 2;
end

switch test_image.type
    case {'CF'}
        true_x = conduct_cf(prior.mesh, test_image.base, test_image.range);
    case {'Inclusion'}
        true_x = conduct_inclusion(prior.mesh, test_image.base, test_image.range);
    case {'Prior'}
        if exist(output.image_str,'file')
            load(output.image_str, 'true_x');
        else
            true_x = conduct_prior(prior);
            save(output.image_str, 'true_x');
        end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cond = conduct_prior(prior)

r       = randn(prior.DoF, 1);
u       = matvec_prior_L(prior, r) + prior.mean_u;
cond    = u2x(prior, u);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cond = conduct_cf(mesh, base, range)

ref = [0.75, 0.25];
r = ref(1)-ref(2);
c = 0.05+1e-10;

alphas = linspace(0,pi/2,20);

ind = false(mesh.Nnode,1);
for i = 1:length(alphas)
    x = ref(1) - r*cos(alphas(i));
    y = ref(2) + r*sin(alphas(i));
    
    j = mesh.node(1,:)>(x-c) & mesh.node(1,:)<(x+c) & mesh.node(2,:)>(y-c) & mesh.node(2,:)<(y+c) ;
    ind = ind | j(:);
end

c = 0.1+1e-10;
ref = [0.7 0.3];
j = mesh.node(1,:)>(ref(1)-c) & mesh.node(1,:)<(ref(1)+c) & mesh.node(2,:)>(ref(2)-c) & mesh.node(2,:)<(ref(2)+c);
ind = ind | j(:);

cond = ones(mesh.Nnode,1)*(base+range);
cond(ind) = base;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cond = conduct_inclusion(mesh, base, range)
% CONDUCT_INCLUSION  generates the test image for EIT, with inclusion
%
% Tiangang Cui, 03/May/2012

centers     = [0.3, 0.5; 0.7 0.5];
r           = [0.1, 0.1];
type        = [2, 2];

ind = false(mesh.Nnode,1);
for i = 1:size(centers,1)
    
    switch type(i)
        case {1}
            j = abs(mesh.node(1,:)-centers(i,1)) < r(i) & abs(mesh.node(2,:)-centers(i,2)) < r(i) ;
        case {2}
            j = ( (mesh.node(1,:)-centers(i,1)).^2 + (mesh.node(2,:)-centers(i,2)).^2 ) < r(i)^2 ;
    end
    ind = ind | j(:);
end

cond = ones(mesh.Nnode,1)*(base+range);
cond(ind) = base;

%%%%%%%%%

%{
high = -5;
low = -3;

ref = [0.7, 0.4];
r = 0.1;
c = 0.02+1e-10;

alphas = linspace(pi/2,pi,20);

ind = false(mesh.N_el,1);
for i = 1:length(alphas)
    x = ref(1) - r*cos(alphas(i));
    y = ref(2) + r*sin(alphas(i));
    
    j = mesh.node(1,:)>(x-c) & mesh.node(1,:)<(x+c) & mesh.node(2,:)>(y-c) & mesh.node(2,:)<(y+c) ;
    ind = ind | j(:);
end

c = 0.02+1e-10;
j = mesh.node(1,:)>(ref(1)-c) & mesh.node(1,:)<(ref(1)+c) & mesh.node(2,:)>(ref(2)-c) & mesh.node(2,:)<(ref(2)+c);
ind = ind | j(:);

cond(ind) = cond(ind) + low;


%%%%%%%%%%%%%%%%

ref = [0.3, 0.3];
r = 0.1;
c = 0.02+1e-10;

alphas = linspace(-pi/2,0,20);

ind = false(mesh.N_el,1);
for i = 1:length(alphas)
    x = ref(1) - r*cos(alphas(i));
    y = ref(2) + r*sin(alphas(i));
    
    j = mesh.node(1,:)>(x-c) & mesh.node(1,:)<(x+c) & mesh.node(2,:)>(y-c) & mesh.node(2,:)<(y+c) ;
    ind = ind | j(:);
end

c = 0.03+1e-10;
j = mesh.node(1,:)>(ref(1)-c) & mesh.node(1,:)<(ref(1)+c) & mesh.node(2,:)>(ref(2)-c) & mesh.node(2,:)<(ref(2)+c);
ind = ind | j(:);

cond(ind) = cond(ind) + low;


%%%%%%%%%%%%%%%%

ref = [0.1, 0.9];
r = 0.1;
c = 0.02+1e-10;

alphas = linspace(pi,pi+pi/2,20);

ind = false(mesh.N_el,1);
for i = 1:length(alphas)
    x = ref(1) - r*cos(alphas(i));
    y = ref(2) + r*sin(alphas(i));
    
    j = mesh.node(1,:)>(x-c) & mesh.node(1,:)<(x+c) & mesh.node(2,:)>(y-c) & mesh.node(2,:)<(y+c) ;
    ind = ind | j(:);
end

c = 0.02+1e-10;
j = mesh.node(1,:)>(ref(1)-c) & mesh.node(1,:)<(ref(1)+c) & mesh.node(2,:)>(ref(2)-c) & mesh.node(2,:)<(ref(2)+c);
ind = ind | j(:);

cond(ind) = cond(ind) + low;

%%%%%%%%%%%%%%%%%%%%

%cond_add = exp(dist_random(prior, 1)/3+0.1);
%cond = cond.*cond_add;

cond = exp(cond);
%}
end