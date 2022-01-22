function boundary_eq = implicit_func3(y_temp,alpha,x,y,s,sig,x_plot,k,N,c,string)

boundary_eq=0;
for i=1:N
    boundary_eq=boundary_eq+ alpha(i)*s(i)* kernel([x(i);y(i)],[x_plot(k);y_temp],string,sig);
end
boundary_eq=boundary_eq-c+1;

% diff=[x y] - ones(N,1)*[x_plot(k) y_temp];
% boundary_eq=sum( alpha.* s .* exp( -1/(2*sig^2)*(diff(:,1).^2+diff(:,2).^2) ) ) -c + 1;
