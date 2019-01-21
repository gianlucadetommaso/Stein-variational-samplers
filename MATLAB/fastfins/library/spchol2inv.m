function iA = spchol2inv(R)
% TC, my implementation

I = speye(size(R));
iR = full(R\I);
iA = iR*iR';

