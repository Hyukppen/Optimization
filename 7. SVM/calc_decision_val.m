function decision_val = calc_decision_val(x_plot,y_plot,alpha,s,x,y,sig,N,c,which_kerne)

decision_val=0;
for i=1:N
    decision_val=decision_val+ alpha(i)*s(i)* kernel([x(i);y(i)],[x_plot;y_plot],which_kerne,sig);
end
decision_val=decision_val-c;
