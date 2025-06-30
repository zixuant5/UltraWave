function [j] = sbessely(n,x)
% x in column, n in row
% j=sqrt(pi./(2*(x*ones(size(n))))).*bessely(n+0.5,x);
j = zeros(length(x),length(n));
    for m=1:length(x)
        j(m,:) = sqrt(pi/(2*x(m)))*bessely(n+0.5,x(m));
    end
end