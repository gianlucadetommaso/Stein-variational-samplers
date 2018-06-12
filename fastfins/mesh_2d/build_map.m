function map = build_map(mesh, mesh_p)

if mesh.N_el < mesh_p.N_el
    % coarse to fine
    map.type = -1;
    [map.p2m, map.m2p] = map_grid(mesh_p, mesh);
elseif mesh.N_el > mesh_p.N_el
    % fine to coarse
    map.type = 1;
    [map.m2p, map.p2m] = map_grid(mesh, mesh_p);
else
    map.type = 0;
end

end

function [f2c, c2f] = map_grid(mesh_f, mesh_c)

fi = zeros(mesh_f.N_el,1);
ci = zeros(mesh_f.N_el,1);
k = 0;
for i = 1:mesh_c.N_el
    ind = mesh_f.centers(1,:) > mesh_c.node(1,mesh_c.node_map(1,i)) ...
        & mesh_f.centers(1,:) < mesh_c.node(1,mesh_c.node_map(3,i)) ...
        & mesh_f.centers(2,:) > mesh_c.node(2,mesh_c.node_map(1,i)) ...
        & mesh_f.centers(2,:) < mesh_c.node(2,mesh_c.node_map(3,i)) ;
    
    tmp = find(ind);
    jnd = k+(1:length(tmp));
    k = k + length(tmp);
    fi(jnd) = tmp;
    ci(jnd) = i;    
end

c2f = sparse(fi, ci, ones(mesh_f.N_el,1), mesh_f.N_el, mesh_c.N_el);
f2c = sparse(ci, fi, ones(mesh_f.N_el,1)*(mesh_c.N_el/mesh_f.N_el), mesh_c.N_el, mesh_f.N_el);
end