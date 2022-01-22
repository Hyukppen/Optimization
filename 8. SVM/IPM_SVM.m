clear
close all
%% SVM - Primal-Dual IPM
% N=20;
% x=[randn(N/2,1) ; randn(N/2,1)+5];
% y=[randn(N/2,1)+5 ; randn(N/2,1)];
% s=[1*ones(N/2,1) ; -1*ones(N/2,1)];
% 
% % Nh=20;
% % Ns=2;
% % x=[randn(Nh/2,1) ; randn(Nh/2,1)+5 ; 3 ; 2];
% % y=[randn(Nh/2,1)+5 ; randn(Nh/2,1) ; 2 ; 2];
% % s=[1*ones(Nh/2,1) ; -1*ones(Nh/2,1); 1 ; -1];
% % N=Nh+Ns;
% 
% mu_ini=1*ones(N,1);
% wk=[-1 ; 1 ; 0 ; mu_ini];
% tl=1;
% 
% figure
% plot(x(s==1),y(s==1),'bo','markersize',15,'linewidth',2)
% hold on; grid on
% plot(x(s==-1),y(s==-1),'ro','markersize',15,'linewidth',2)
% axis([-2 6 -4 10])
% x_plot=-2:0.01:6;
% 
% for k=1:50
%     tl1=0.7*tl;
%     t=tl1;
%     
%     a=wk(1);
%     b=wk(2);
%     c=wk(3);
%     mu=wk(4:end);
%     g=s.*(c - [a b] * [x.' ; y.']).'+1;
%     dgdx = [-s.*x -s.*y s];
%     
%     R=[ [a ; b ; 0]+ dgdx.'*mu ; mu.*g + t*ones(N,1) ];
%     B1 = [1 0 0 ; 0 1 0 ; 0 0 0];
%     B2 = dgdx.';
%     B3 = diag(mu)*dgdx;
%     B4 = diag(g);
%     dRdw=[ B1 B2 ; B3 B4 ];
%     
%     wk1=wk-inv(dRdw)*R;
%     wk=wk1;
%     tl=tl1;
% 
%     hold off
%     plot(x(s==1),y(s==1),'bo','markersize',15,'linewidth',2)
%     hold on; grid on
%     plot(x(s==-1),y(s==-1),'ro','markersize',15,'linewidth',2)
%     plot(x_plot,-a/b*x_plot+c/b,'k','linewidth',2)
%     plot(x_plot,-a/b*x_plot+(c+1)/b,'r','linewidth',2)
%     plot(x_plot,-a/b*x_plot+(c-1)/b,'r','linewidth',2)
%     plot(x(abs(mu)>0.001),y(abs(mu)>0.001),'pg','markersize',10,'markerfacecolor','g')
%     axis([-2 6 -4 10])
%     pause(0.1)
% end
% g

%% Soft margin SVM - Primal-Dual IPM
% Nh=20;
% Ns=2;
% x=[randn(Nh/2,1) ; randn(Nh/2,1)+5 ; 3 ; 2];
% y=[randn(Nh/2,1)+5 ; randn(Nh/2,1) ; 2 ; 2];
% s=[1*ones(Nh/2,1) ; -1*ones(Nh/2,1); 1 ; -1];
% N=Nh+Ns;
% 
% eps_ini=[1*ones(Nh,1) ; 100*ones(Ns,1)]; % for feasible start (phase 1)
% mu_ini=1*ones(2*N,1);
% wk=[-1 ; 1 ; 0 ; eps_ini ; mu_ini];
% 
% gamma=0.1;
% % gamma=1;
% % gamma=10;
% tl=1;
% 
% figure
% plot(x(s==1),y(s==1),'bo','markersize',15,'linewidth',2)
% hold on; grid on
% plot(x(s==-1),y(s==-1),'ro','markersize',15,'linewidth',2)
% axis([-2 6 -2 8])
% x_plot=-2:0.01:6;
% 
% for k=1:50
%     tl1=0.8*tl;
%     t=tl1;
%     
%     a=wk(1);
%     b=wk(2);
%     c=wk(3);
%     eps=wk(4:4+N-1);
%     mu=wk(3+N+1:end);
%     g=[s.*(c - [a b] * [x.' ; y.']).'+1 - eps ; -eps];
%     dgdx = [ [-s.*x -s.*y s -1*eye(N)] ; [zeros(N,3) -1*eye(N)] ];
%     
%     R=[ [a ; b ; 0; gamma*ones(N,1)] + dgdx.'*mu ; mu.*g + t*ones(2*N,1) ];
%     B1 = [1 0 0 ; 0 1 0 ; 0 0 0];
%     B1(N+3,N+3) = 0;
%     B2 = dgdx.';
%     B3 = diag(mu)*dgdx;
%     B4 = diag(g);
%     dRdw=[ B1 B2 ; B3 B4 ];
%     
%     wk1=wk-inv(dRdw)*R;
%     wk=wk1;
%     tl=tl1;
% 
%     hold off
%     plot(x(s==1),y(s==1),'bo','markersize',15,'linewidth',2)
%     hold on; grid on
%     plot(x(s==-1),y(s==-1),'ro','markersize',15,'linewidth',2)
%     plot(x_plot,-a/b*x_plot+c/b,'k','linewidth',2)
%     plot(x_plot,-a/b*x_plot+(c+1)/b,'r','linewidth',2)
%     plot(x_plot,-a/b*x_plot+(c-1)/b,'r','linewidth',2)
%     plot(x(abs(mu(1:N))>0.001),y(abs(mu(1:N))>0.001),'pg','markersize',10,'markerfacecolor','g')
%     axis([-2 6 -2 8])
%     pause(0.1)
% end
% 
% g(1:N)
% alpha=mu(1:N)
% beta=mu(N+1:end)
%% SVM_kernel trick - Dual problem using Primal-Dual IPM
Nh=20;
Ns=2;
x=[randn(Nh/2,1) ; randn(Nh/2,1)+5 ; 3 ; 2];
y=[randn(Nh/2,1)+5 ; randn(Nh/2,1) ; 2 ; 2];
s=[1*ones(Nh/2,1) ; -1*ones(Nh/2,1); 1 ; -1];
N=Nh+Ns;

alpha_ini=1*ones(N,1);
beta_ini=1*ones(N,1);
mu_ini=1*ones(2*N,1);
lam_ini=1*ones(N+1,1);
wk=[alpha_ini ; beta_ini ; mu_ini ; lam_ini];

gamma=1;
sig=1;
step_size=1;
tl=1;

figure
plot(x(s==1),y(s==1),'bo','markersize',15,'linewidth',2)
hold on; grid on
plot(x(s==-1),y(s==-1),'ro','markersize',15,'linewidth',2)
axis([-2 6 -2 8])
x_plot=-2:0.01:6;

for k=1:50
    tl1=0.8*tl;
    t=tl1;
    
    alpha=wk(1:N);
    beta=wk(N+1:2*N);
    mu=wk(2*N+1:4*N);
    lam=wk(4*N+1:end);

    g=[-alpha ; -beta];
    h=[sum(alpha.*s) ; alpha+beta-gamma];
    dgdx = -eye(2*N);
    dhdx = [s.' zeros(1,N) ; eye(N) eye(N)];
    
    df=zeros(N,1);
    for j=1:N
        for i=1:N
            df(j)= df(j)+alpha(i)*s(i)*s(j)* kernel([x(i);y(i)],[x(j);y(j)],'Linear',sig);
        end
        df(j)= df(j)-1;
    end
    d2f=zeros(N,N);
    for row=1:N
        for col=1:N
            d2f(row,col) = s(row)*s(col)* kernel([x(row);y(row)],[x(col);y(col)],'Linear',sig);
        end
    end
    R=[ [df ; zeros(N,1)] + dhdx.'*lam + dgdx.'*mu ; mu.*g + t*ones(2*N,1) ; h ];
    B11 = d2f;
    B11(2*N,2*N) = 0;
    B12 = dgdx.';
    B13 = dhdx.';
    B21 = diag(mu)*dgdx;
    B22 = diag(g);
    B23 = zeros(2*N,N+1);
    B31 = dhdx;
    B32 = zeros(N+1,2*N);
    B33 = zeros(N+1,N+1);
    dRdw=[ B11 B12 B13 ; B21 B22 B23 ; B31 B32 B33 ];
    
    wk1=wk-step_size*inv(dRdw)*R; %%%%%%%%%%%%%%%%%%%%%% step_size for stable update
    wk=wk1;
    tl=tl1;
    
end


[~, support_j]=max(abs(alpha));
% support_i=1
c=zeros(1);
for i=1:N
    c=c+ alpha(i)*s(i)* kernel([x(i);y(i)],[x(support_j);y(support_j)],'Linear',sig);
end
c=c-s(support_j);
% c=0;


y_plot=-100:0.1:100;
boundary_eq=zeros(length(y_plot),1);
for k=1:length(y_plot)
    for i=1:N
        boundary_eq(k)=boundary_eq(k)+ alpha(i)*s(i)* kernel([x(i);y(i)],[x_plot(1);y_plot(k)],'Linear',sig);
    end
    boundary_eq(k) = boundary_eq(k) -c;
end
figure
plot(y_plot,boundary_eq)


y_plot1=x_plot;
y_plot2=x_plot;
y_plot3=x_plot;
for k=1:length(x_plot)
    y_plot1(k)=fzero(@(y_temp) implicit_func1(y_temp,alpha,x,y,s,sig,x_plot,k,N,c,'Linear'), x_plot(k));
    y_plot2(k)=fzero(@(y_temp) implicit_func2(y_temp,alpha,x,y,s,sig,x_plot,k,N,c,'Linear'), x_plot(k));
    y_plot3(k)=fzero(@(y_temp) implicit_func3(y_temp,alpha,x,y,s,sig,x_plot,k,N,c,'Linear'), x_plot(k));
end
figure
hold off
plot(x_plot,y_plot1,'k','linewidth',2); hold on; grid on
plot(x_plot,y_plot2,'r','linewidth',2)
plot(x_plot,y_plot3,'r','linewidth',2)
plot(x(s==1),y(s==1),'bo','markersize',15,'linewidth',2)
plot(x(s==-1),y(s==-1),'ro','markersize',15,'linewidth',2)
plot(x(abs(alpha)>0.001),y(abs(alpha)>0.001),'pg','markersize',10,'markerfacecolor','g')
axis([-2 6 -2 8])

%% IPM - min x^2 s.t. 1<=x<=3
% close all
% 
% mu_ini=[1 ; 1];
% wk=[2 ; mu_ini];
% tl=1;
% 
% x_plot=-5:0.01:5;
% figure
% plot(x_plot,x_plot.^2); hold on; grid on;
% plot(wk(1),wk(1).^2,'o','markersize',8,'markerfacecolor','r','markeredgecolor','r')
% text(-4,20,'iterations = 0','fontsize',15)
% pause(0.2)
% 
% for k=1:100
%     tl1=0.8*tl;
%     t=tl1;
%     
%     x=wk(1);
%     mu=wk(2:3);
%     dgdx=[-1 ; 1];
%     g = [-x+1; x-3];
%     
%     R=[ 2*x+ dgdx.'*mu ; mu.*g + t*ones(2,1)];
%     dRdw=[ 2 dgdx.' ; diag(mu)*dgdx diag(g) ];
%     
%     wk1=wk-inv(dRdw)*R;
%     wk=wk1;
%     tl=tl1;
%     
%     hold off
%     plot(x_plot,x_plot.^2); hold on; grid on;
%     plot(wk1(1),wk1(1)^2,'o','markersize',8,'markerfacecolor','r','markeredgecolor','r')
%     text(-4,20,['iterations = ', num2str(k)],'fontsize',15)
%     pause(0.2)
% end

%% IPM - min x^3 s.t. 1<=x<=3
% close all
% 
% mu_ini=[1 ; 1];
% wk=[2 ; mu_ini];
% tl=1;
% 
% x_plot=-5:0.01:5;
% figure
% plot(x_plot,x_plot.^3); hold on; grid on;
% plot(wk(1),wk(1).^2,'o','markersize',8,'markerfacecolor','r','markeredgecolor','r')
% text(-4,20,'iterations = 0','fontsize',15)
% pause(0.01)
% 
% for k=1:100
%     tl1=0.8*tl;
%     t=tl1;
%     
%     x=wk(1);
%     mu=wk(2:3);
%     dgdx=[-1 ; 1];
%     g = [-x+1; x-3];
%     
%     R=[ 3*x^2 + dgdx.'*mu ; mu.*g + t*ones(2,1)];
%     dRdw=[ 6*x dgdx.' ; diag(mu)*dgdx diag(g) ];
%     
%     wk1=wk-inv(dRdw)*R;
%     wk=wk1;
%     tl=tl1;
%     
%     hold off
%     plot(x_plot,x_plot.^3); hold on; grid on;
%     plot(wk1(1),wk1(1)^2,'o','markersize',8,'markerfacecolor','r','markeredgecolor','r')
%     text(-4,20,['iterations = ', num2str(k)],'fontsize',15)
%     pause(0.01)
% end
