function [map_f2c, map_c2f] = build_bimap(mesh_coarse, mesh_fine)

s = mesh_fine.Nel/mesh_coarse.Nel;
is = zeros(mesh_fine.Nel, 1);
js = zeros(mesh_fine.Nel, 1);
for i = 1:mesh_coarse.Nel
    ind = mesh_coarse.node_map(:,i);
    % bounded by first and third point
    lb = mesh_coarse.node(:, ind(1));
    ru = mesh_coarse.node(:, ind(3));
    jj = find(  mesh_fine.centers(1,:) < ru(1) & ...
                mesh_fine.centers(1,:) > lb(1) & ...
                mesh_fine.centers(2,:) < ru(2) & ...
                mesh_fine.centers(2,:) > lb(2) );
    if length(jj) ~= s
        disp('matching error')
    end
    ii = s*(i-1) + (1:s);
    is(ii) = i;
    js(ii) = jj;
end

tmp = sparse(is, js, ones(mesh_fine.Nel, 1), mesh_coarse.Nel, mesh_fine.Nel);
% averaging
map_f2c = tmp/s;
% refine
map_c2f = tmp';

end