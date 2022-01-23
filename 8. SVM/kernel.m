function out = kernel(x,y,which_kernel,sig)

if strcmp(which_kernel,'Linear')
    out = x.'*y;
elseif strcmp(which_kernel,'poly')
    a=.1; c=0; d=5;
    out = (a*x.'*y+c)^d;
elseif strcmp(which_kernel,'RBF')
    out = exp(-1/(2*sig^2) * sum((x-y).^2) );
elseif strcmp(which_kernel,'sigmoid')
    out = tanh(0.1*x.'*y-1);
end
