figure
%%
x1=-10:0.1:10;
x2=x1;
[X1,X2]=meshgrid(x1,x2);
lam=-0.1;
f=X1+X2+lam*(X1.^2+X2.^2-1);
surf(X1,X2,f)
shading flat

%%
t=0.1;
gx=linspace(-1,-eps,10000);
f=-t*log(-gx);

figure
plot(gx,f)
t=2;
f=-t*log(-gx);
hold on
plot(gx,f)


%%

x1=-4:0.1:4;
x2=-3:0.1:3;
[X1,X2]=meshgrid(x1,x2);

f=X1.^2+10*X2.^2;

init=[2;2];
alpha=0.1;
xk_grad(:,1)=init;
for k=1:10
    xk_grad(:,k+1)=xk_grad(:,k)-alpha*[2*xk_grad(1,k) ; 20*xk_grad(2,k)];
end


xk_Newton(:,1)=init;
for k=1:3
    xk_Newton(:,k+1)=xk_Newton(:,k)-inv([2 0 ; 0 20])*[2*xk_Newton(1,k) ; 20*xk_Newton(2,k)];
end

figure
contour(X1,X2,f)
axis image
hold on
plot(xk_grad(1,:),xk_grad(2,:),'k.-','markersize',20)
plot(xk_Newton(1,:),xk_Newton(2,:),'r.-','markersize',20)
legend('minimize x1^2+10*x2^2','xk\_gradient','xk\_Newton','location','northwest')
xlabel('x1')
ylabel('x2')
title('both starts at [2, 2]')

%% 
x1=-3:0.1:3;
x2=x1;
[X1,X2]=meshgrid(x1,x2);
f=X1.^2+X2;

figure
surf(X1,X2,f)
shading flat


%% interior point method - cost function
x1=linspace(-1.2,1.2,100);
x2=linspace(-1.2,1.2,100);
[X1,X2]=meshgrid(x1,x2);

t=100;
f=X1+X2;
g=X1.^2+X2.^2-1;
L = f - t*log(-g);

L(imag(L)~=0)=max(max(real(L)));
L=L-min(min(L));
L=L/max(max(L));

figure('position',[-800 210 500 900])
surf(X1,X2,L)
xlabel('x1')
ylabel('x2')
% axis image
xlim([min(x1) max(x1)])
ylim([min(x2) max(x2)])
zlim([0 1])
shading interp
lightangle(65,10)
view(50,5)
%% interior point method - cost function (video)
x1=linspace(-1.2,1.2,100);
x2=linspace(-1.2,1.2,100);
[X1,X2]=meshgrid(x1,x2);

t=0.01;
f=X1+X2;
g=X1.^2+X2.^2-1;

figure('position',[-800 210 500 900])
high=0;
low=1;
while(1)
    if t > 30
        high=1;
        low=0;
    end
    if t < 0.01
        low=1;
        high=0;
    end
    if high==1
        t=0.9*t;
    end
    if low==1
        t=1.1*t;
    end
    L = f - t*log(-g);
    L(imag(L)~=0)=max(max(real(L)));
    L=L-min(min(L));
    L=L/max(max(L));
    surf(X1,X2,L)
    xlabel('x1')
    ylabel('x2')
    % axis image
    xlim([min(x1) max(x1)])
    ylim([min(x2) max(x2)])
    zlim([0 1])
    shading interp
    lightangle(65,10)
    view(50,5)
    pause(0.01)
end
%% interior point method - IPM
xk=[0;0];
tl=1;
for l=1:50
    tl1=0.8*tl;
    for k=1:5
        x1=xk(1);
        x2=xk(2);
        t=tl1;
        g=[ 1+t*(2*x1)/(-x1^2-x2^2+1) ; 1+t*(2*x2)/(-x1^2-x2^2+1) ];
        H=[t*(2*x1^2-2*x2^2+2)/(1-x1^2-x2^2)^2 t*4*x1*x2/(1-x1^2-x2^2)^2 ; ...
            t*4*x1*x2/(1-x1^2-x2^2)^2           t*(-2*x1^2+2*x2^2+2)/(1-x1^2-x2^2)^2];
        
        xk1=xk-inv(H)*g;
        xk=xk1;
    end
    tl=tl1;
end

%% interior point method - Primal-Dual IPM
wk=[0 ; 0 ; 1];
tl=1;
for k=1:45
    tl1=0.8*tl;
    
    x1=wk(1);
    x2=wk(2);
    mu=wk(3);
    
    t=tl1;
    R=[ 1+2*x1*mu ; 1+2*x2*mu ; mu*(x1^2+x2^2-1)+t ];
    dRdw=[ 2           0         2*x1 ; ...
           0           2         2*x2 ; ...
           2*mu*x1  2*mu*x2   x1^2+x2^2-1 ];
    
    wk1=wk-inv(dRdw)*R;
    wk=wk1;
    tl=tl1;
end


%% interior point method - IPM (for plot)
xk=[0;0];
tl=1;

x1_plot=linspace(-1.2,1.2,100);
x2_plot=linspace(-1.2,1.2,100);
[X1,X2]=meshgrid(x1_plot,x2_plot);
f=X1+X2;
g=X1.^2+X2.^2-1;
t=1;
L = f - t*log(-g);
L(imag(L)~=0)=max(max(real(L)));
figure('position',[-800 250 900 900])
contour(x1_plot,x2_plot,L,30)
axis image
xlabel('x1')
ylabel('x2')
grid on
hold on
plot(xk(1),xk(2),'ko','markersize',8,'markerfacecolor','k','markeredgecolor','k')
hold off
x_plot=xk;
pause(0.5)

for l=1:50
    tl1=0.8*tl;
    
    L = f - tl1*log(-g);
    L(imag(L)~=0)=max(max(real(L)));
    hold off
    contour(x1_plot,x2_plot,L,30); axis([-1 0.1 -1 0.1]); grid on; hold on; axis image
    pause(0.1)
    for k=1:6
        x1=xk(1);
        x2=xk(2);
        t=tl1;
        grad=[ 1+t*(2*x1)/(-x1^2-x2^2+1) ; 1+t*(2*x2)/(-x1^2-x2^2+1) ];
        H=[t*(2*x1^2-2*x2^2+2)/(1-x1^2-x2^2)^2 t*4*x1*x2/(1-x1^2-x2^2)^2 ; ...
            t*4*x1*x2/(1-x1^2-x2^2)^2           t*(-2*x1^2+2*x2^2+2)/(1-x1^2-x2^2)^2];
        
        xk1=xk-inv(H)*grad;
        xk=xk1;
        x_plot=[x_plot xk];
        plot(x_plot(1,:),x_plot(2,:),'ko--','markersize',8,'markerfacecolor','k','markeredgecolor','k')
        pause(0.01)
    end
    tl=tl1;
    
    x_plot=xk;
    plot(x_plot(1,:),x_plot(2,:),'rp','markersize',15,'markerfacecolor','r','markeredgecolor','r')
    pause(0.1)
end

%% interior point method - Primal-Dual IPM (for plot)
wk=[0 ; 0 ; 1];
tl=1;

x1_plot=linspace(-1.2,1.2,100);
x2_plot=linspace(-1.2,1.2,100);
[X1,X2]=meshgrid(x1_plot,x2_plot);
f=X1+X2;
g=X1.^2+X2.^2-1;
t=1;
L = f - t*log(-g);
L(imag(L)~=0)=max(max(real(L)));
figure('position',[-800 250 900 900])
contour(x1_plot,x2_plot,L,30)
axis image
xlabel('x1')
ylabel('x2')
grid on
hold on
plot(wk(1),wk(2),'ko','markersize',8,'markerfacecolor','k','markeredgecolor','k')
hold off
x_plot=wk(1:2);

for k=1:45
    tl1=0.8*tl;
    
    L = f - tl1*log(-g);
    L(imag(L)~=0)=max(max(real(L)));
    hold off
    contour(x1_plot,x2_plot,L,30); axis([-1 0.1 -1 0.1]); grid on; hold on; axis image
    
    x1=wk(1);
    x2=wk(2);
    mu=wk(3);
    
    t=tl1;
    R=[ 1+2*x1*mu ; 1+2*x2*mu ; mu*(x1^2+x2^2-1)+t ];
    dRdw=[ 2           0         2*x1 ; ...
           0           2         2*x2 ; ...
           2*mu*x1  2*mu*x2   x1^2+x2^2-1 ];
    
    wk1=wk-inv(dRdw)*R;
    wk=wk1;
    tl=tl1;
    
    x_plot=[x_plot wk(1:2)];
    plot(x_plot(1,:),x_plot(2,:),'ro--','markersize',8,'markerfacecolor','r','markeredgecolor','r')
    pause(0.1)
end


