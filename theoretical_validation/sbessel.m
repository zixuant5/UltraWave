function [j] = sbessel(n,x)
% x in column, n in row
% j=sqrt(pi./(2*(x*ones(size(n))))).*besselj(n+0.5,x);
j = zeros(length(x),length(n));
    for m=1:length(x)
        j(m,:) = sqrt(pi/(2*x(m)))*besselj(n+0.5,x(m));
    end
end