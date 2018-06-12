function f = triangle_func(x, left, right, type)

tol = 1E-10;
ind = x < left-tol | x > right+tol;

switch type
    case {'right'}
        f = (x - left) / (right - left);
    case {'center'}
        mid = (left+right)*0.5;
        il = x < mid;
        ir = x >= mid; 
        f(il) = (x(il) - left) / (mid - left);
        f(ir) = (x(ir) - right) / (mid - right);
    case {'left'}
        f = (x - right) / (left - right);
    otherwise
        f = zeros(size(x));
end

f(ind) = 0;

end