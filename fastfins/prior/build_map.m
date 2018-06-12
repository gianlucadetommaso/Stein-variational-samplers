function map = build_map(mesh_to, mesh_from)

s = mesh_from.Nel/mesh_to.Nel;
is = zeros(mesh_from.Nel, 1);
js = zeros(mesh_from.Nel, 1);
for i = 1:mesh_to.Nel
    ind = mesh_to.node_map(:,i);
    % bounded by first and third point
    lb = mesh_to.node(:, ind(1));
    ru = mesh_to.node(:, ind(3));
    jj = find(  mesh_from.centers(1,:) < ru(1) & ...
                mesh_from.centers(1,:) > lb(1) & ...
                mesh_from.centers(2,:) < ru(2) & ...
                mesh_from.centers(2,:) > lb(2) );
    if length(jj) ~= s
        disp(length(jj)), disp(s)
        disp('matching error')
    end
    ii = s*(i-1) + (1:s);
    is(ii) = i;
    js(ii) = jj;
end

map = sparse(is, js, ones(mesh_from.Nel, 1), mesh_to.Nel, mesh_from.Nel)/s;

end