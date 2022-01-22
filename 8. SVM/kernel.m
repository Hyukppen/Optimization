function out = kernel(x,y,string,sig)

if strcmp(string,'Linear')
    out = x.'*y;
elseif strcmp(string,'RBF')
    out = exp(-1/(2*sig^2) * sum((x-y).^2) );
end
