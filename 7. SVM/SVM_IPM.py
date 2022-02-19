import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import time
# %% SVM - Primal-Dual IPM
plt.close('all') 

Nh=20; Ns=0; N=Nh+Ns
r1=np.random.randn(int(Nh/2),1); r2=np.random.randn(int(Nh/2),1)+5
s1=np.ones((int(Nh/2),1)); s2=-1*np.ones((int(Nh/2),1))
# case 1
x=np.vstack((r1,r2))
y=np.vstack((r2,r1))
s=np.vstack((s1,s2))
xmin= -3; xmax= 8; ymin= -3; ymax = 8
# case 2
# Nh=20; Ns=2; N=Nh+Ns
# x=np.vstack((r1,r2, 3, 2))
# y=np.vstack((r2,r1, 2, 2))
# s=np.vstack((s1,s2, 1, -1))
# xmin= -3; xmax= 8; ymin= -3; ymax = 8

mu_ini=1*np.ones((N,1))
wk=np.vstack((-1,1,0,mu_ini))
tl=1;

figure = plt.figure(figsize=[8.5, 7.5])

plt.grid()
plt.plot(x[s==-1],y[s==-1],'ro',fillstyle="none",markersize=15,markeredgewidth=2)
plt.plot(x[s==1],y[s==1],'bo',fillstyle="none",markersize=15,markeredgewidth=2)
plt.axis([xmin, xmax, ymin, ymax])
x_plot=np.linspace(xmin,xmax,100)

for k in range(50):
    tl1=0.8*tl
    t=tl1
    
    a=wk[0].reshape(1,1)
    b=wk[1].reshape(1,1)
    c=wk[2].reshape(1,1)
    mu=wk[3:]
    
    ab=np.hstack((a,b))
    xy=np.vstack((np.transpose(x), np.transpose(y)))
    g=s*np.transpose(c-ab@xy)+1
    
    dgdx = np.hstack((-s*x, -s*y, s))
    
    R1 = np.vstack((a,b,0)) + np.transpose(dgdx)@mu
    R2 = mu*g + t*np.ones((N,1))
    R = np.vstack((R1,R2))
    
    B1 = np.array([ [1, 0, 0], [0, 1, 0], [0, 0, 0] ])
    B2 = np.transpose(dgdx)
    B3 = np.diag(np.squeeze(mu))@dgdx
    B4 = np.diag(np.squeeze(g))
    dRdw=np.block([ [B1,B2] , [B3,B4] ])
    
    wk1=wk-np.linalg.inv(dRdw)@R
    
    wk=wk1
    tl=tl1

    plt.cla()
    
    plt.plot(x[s==1],y[s==1],'bo',fillstyle="none",markersize=15,markeredgewidth=2)
    plt.grid()
    plt.plot(x[s==-1],y[s==-1],'ro',fillstyle="none",markersize=15,markeredgewidth=2)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.plot(x_plot,np.squeeze(-a/b*x_plot+c/b),'k',linewidth=2)
    plt.plot(x_plot,np.squeeze(-a/b*x_plot+(c+1)/b),'r',linewidth=2)
    plt.plot(x_plot,np.squeeze(-a/b*x_plot+(c-1)/b),'r',linewidth=2)
    plt.plot(x[abs(mu)>0.001],y[abs(mu)>0.001],'*g',markersize=10,markerfacecolor='g')
    
    figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(0.01)

print(g)

# %% Soft margin SVM - Primal-Dual IPM
plt.close('all')
Nh=20; Ns=2; N=Nh+Ns
r1=np.random.randn(int(Nh/2),1); r2=np.random.randn(int(Nh/2),1)+5
s1=np.ones((int(Nh/2),1)); s2=-1*np.ones((int(Nh/2),1))

x=np.vstack((r1,r2, 3, 2))
y=np.vstack((r2,r1, 2, 2))
s=np.vstack((s1,s2, 1, -1))
xmin= -3; xmax= 8; ymin= -3; ymax = 8

eps_ini=np.vstack( [1*np.ones((Nh,1)), 10*np.ones((Ns,1))] )
mu_ini=1*np.ones((2*N,1))
wk=np.vstack((-1,1,0,eps_ini,mu_ini))

gamma=0.1
# gamma=1
# gamma=10
tl=1

figure = plt.figure(figsize=[8.5, 7.5])
plt.plot(x[s==1],y[s==1],'bo',fillstyle="none",markersize=15,markeredgewidth=2)
plt.grid()
plt.plot(x[s==-1],y[s==-1],'ro',fillstyle="none",markersize=15,markeredgewidth=2)
plt.axis([xmin, xmax, ymin, ymax])
x_plot=np.linspace(xmin,xmax,100)

for k in range(50):
    tl1=0.8*tl
    t=tl1
    
    a=wk[0].reshape(1,1)
    b=wk[1].reshape(1,1)
    c=wk[2].reshape(1,1)
    eps=wk[3:3+N]
    mu=wk[2+N+1:]
    
    ab=np.hstack((a,b))
    xy=np.vstack((np.transpose(x), np.transpose(y)))
    g=np.vstack([s*np.transpose(c-ab@xy)+1 -eps , -eps])
    
    dgdx = np.block([[-s*x, -s*y, s, -1*np.eye(N)], [np.zeros((N,3)), -1*np.eye(N)]])
    
    R1 = np.vstack([a,b,0,gamma*np.ones((N,1)) ]) + np.transpose(dgdx)@mu
    R2 = mu*g + t*np.ones((2*N,1))
    R = np.vstack((R1,R2))
    
    B1 = np.zeros((N+3,N+3)); B1[0,0]=1; B1[1,1]=1
    B2 = np.transpose(dgdx)
    B3 = np.diag(np.squeeze(mu))@dgdx
    B4 = np.diag(np.squeeze(g))
    dRdw=np.block([ [B1,B2] , [B3,B4] ])

    wk1=wk-np.linalg.inv(dRdw)@R
    
    wk=wk1
    tl=tl1
    
    plt.cla()
    
    plt.plot(x[s==1],y[s==1],'bo',fillstyle="none",markersize=15,markeredgewidth=2)
    plt.grid()
    plt.plot(x[s==-1],y[s==-1],'ro',fillstyle="none",markersize=15,markeredgewidth=2)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.plot(x_plot,np.squeeze(-a/b*x_plot+c/b),'k',linewidth=2)
    plt.plot(x_plot,np.squeeze(-a/b*x_plot+(c+1)/b),'r',linewidth=2)
    plt.plot(x_plot,np.squeeze(-a/b*x_plot+(c-1)/b),'r',linewidth=2)
    plt.plot(x[abs(mu[:N])>0.001],y[abs(mu[:N])>0.001],'*g',markersize=10,markerfacecolor='g')
    
    figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(0.01)

print(g[:N])
alph=mu[:N]
print(alph)
beta=mu[N:]
print(beta)
# %% SVM_kernel trick - Dual problem using Primal-Dual IPM
plt.close('all')
########### functions ###########
def kernel(x,y,which_kernel,sig):
    if which_kernel=='Linear':
        out = np.transpose(x)@y
    elif which_kernel=='poly':
        a=.1; c=0; d=5
        out = (a*np.transpose(x)@y+c)**d
    elif which_kernel=='RBF':
        out = np.exp(-1/(2*sig**2) * sum((x-y)**2))
    elif which_kernel=='sigmoid':
        out = np.tanh(0.1*np.transpose(x)@y-1)
    return out
def calc_decision_val(x_plot,y_plot,alph,s,x,y,sig,N,c,which_kernel):
    decision_val=0
    for i in range(N):
        decision_val=decision_val+ alph[i]*s[i]* \
            kernel(np.vstack([x[i],y[i]]),np.vstack([x_plot,y_plot]),which_kernel,sig)
    decision_val=decision_val-c;
    return decision_val
#################################

# case 1
Nh=20; Ns=2; N=Nh+Ns
r1=np.random.randn(int(Nh/2),1); r2=np.random.randn(int(Nh/2),1)+5
s1=np.ones((int(Nh/2),1)); s2=-1*np.ones((int(Nh/2),1))
x=np.vstack((r1,r2, 3, 2))
y=np.vstack((r2,r1, 2, 2))
s=np.vstack((s1,s2, 1, -1))
xmin= -3; xmax= 8; ymin= -3; ymax = 8
# case 2
# Nh1=20; Nh2=5; N=Nh1+Nh2
# radius1=2+0.2*np.random.randn(Nh1,1)
# radius2=0.3+0.1*np.random.randn(Nh2,1)
# theta1=2*np.pi*np.random.rand(Nh1,1)-np.pi
# theta2=2*np.pi*np.random.rand(Nh2,1)-np.pi
# x=np.vstack([radius1*np.cos(theta1) , radius2*np.cos(theta2) ])
# y=np.vstack([radius1*np.sin(theta1) , radius2*np.sin(theta2) ])
# s=np.vstack([1*np.ones((Nh1,1)) , -1*np.ones((Nh2,1))])
# xmin= -2.5; xmax= 2.5; ymin= -2.5; ymax = 2.5
# case 3
# Nh=10; N=Nh;
# x1=np.linspace(-2,2,int(Nh/2)).reshape(int(Nh/2),1); x2=x1
# x=np.vstack([x1 ,x2])
# y1=4+x1**2; y2=x2**2
# y=np.vstack([y1 , y2])
# s=np.vstack([-1*np.ones((int(Nh/2),1)) , 1*np.ones((int(Nh/2),1))])
# xmin= -3; xmax= 3; ymin= -0.5; ymax = 8.5

alpha_ini=1*np.ones((N,1))
beta_ini=1*np.ones((N,1))
mu_ini=1*np.ones((2*N,1))
lam_ini=0*np.ones((N+1,1))
wk=np.vstack([alpha_ini , beta_ini , mu_ini , lam_ini])

gamma= 2 # 10 #  0.1 #  1 # 
sig= 1 # 2 # 0.5 # 
tl=1
which_kernel= 'RBF' # 'Linear' #  'poly' # 'sigmoid' #

figure = plt.figure(figsize=[8.5, 7.5])
plt.plot(x[s==1],y[s==1],'bo',fillstyle="none",markersize=15,markeredgewidth=2)
plt.grid()
plt.plot(x[s==-1],y[s==-1],'ro',fillstyle="none",markersize=15,markeredgewidth=2)
plt.axis([xmin, xmax, ymin, ymax])
x_plot=np.linspace(xmin,xmax,30)
y_plot=np.linspace(ymin,ymax,30)
X_plot, Y_plot=np.meshgrid(x_plot,y_plot)
for k in range(3):
    figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(1)
plt.close()
figure = plt.figure(figsize=[8.5, 7.5])
    
for k in range(50):
    tl1=0.8*tl;
    t=tl1;
    
    alph=wk[:N]
    beta=wk[N:2*N]
    mu=wk[2*N:4*N]
    lam=wk[4*N:]

    g=np.vstack([-alph , -beta])
    h=np.vstack([sum(alph*s) , alph+beta-gamma])
    dgdx = -np.eye(2*N)
    dhdx = np.block([[np.transpose(s), np.zeros((1,N))] , [np.eye(N), np.eye(N)]])
            
    df=np.zeros((N,1))
    for j in range(N):
        for i in range(N):
            df[j]= df[j]+alph[i]*s[i]*s[j]* \
                kernel(np.vstack([x[i],y[i]]),np.vstack([x[j],y[j]]),which_kernel,sig)
        df[j]= df[j]-1
        
    d2f=np.zeros((N,N))
    for row in range(N):
        for col in range(N):
            d2f[row,col] = s[row]*s[col]* \
                kernel(np.vstack([x[row],y[row]]),np.vstack([x[col],y[col]]),which_kernel,sig);

    R1= np.vstack([df,np.zeros((N,1))]) + np.transpose(dhdx)@lam + np.transpose(dgdx)@mu
    R2= mu*g + t*np.ones((2*N,1))
    R3= h
    R= np.vstack([R1,R2,R3])
    B11 = np.zeros((2*N,2*N))
    B11[:N,:N] = d2f
    B12 = np.transpose(dgdx)
    B13 = np.transpose(dhdx)
    B21 = np.diag(np.squeeze(mu))@dgdx
    B22 = np.diag(np.squeeze(g))
    B23 = np.zeros((2*N,N+1))
    B31 = dhdx;
    B32 = np.zeros((N+1,2*N))
    B33 = np.zeros((N+1,N+1))
    dRdw=np.block([ [B11, B12, B13], [B21, B22, B23], [B31, B32, B33] ])
    
    wk1=wk-np.linalg.inv(dRdw)@R
    
    wk=wk1
    tl=tl1
    

    #### plot ####
    support_j=np.argmin(abs(alph-gamma/2))
    c=0
    for i in range(N):
        c=c+ alph[i]*s[i]* \
            kernel(np.vstack([x[i],y[i]]),np.vstack([x[support_j],y[support_j]]),which_kernel,sig)
    c=c-s[support_j]

    decision_val=np.zeros((y_plot.size,x_plot.size))
    for xi in range(x_plot.size):
        for yi in range(y_plot.size):
            decision_val[yi,xi] = calc_decision_val(x_plot[xi],y_plot[yi],alph,s,x,y,sig,N,c,which_kernel)

    
    ax = figure.gca(projection="3d")
    ls = LightSource(azdeg=-130, altdeg=15)
    ax.view_init(elev=15,azim=-130)
    rgb = ls.shade(decision_val, plt.cm.RdYlBu)
    ax.plot_surface(X_plot,Y_plot,decision_val, facecolors=rgb, zorder=0)
    # ax.plot_surface(X_plot,Y_plot,decision_val,rstride=1, cstride=1, linewidth=0,
                            # antialiased=False, facecolors=rgb, zorder=0)
    plt.plot(x[s==1],y[s==1],'bo',fillstyle="none",markersize=15,markeredgewidth=2,zorder=100)
    plt.plot(x[s==-1],y[s==-1],'ro',fillstyle="none",markersize=15,markeredgewidth=2,zorder=100)
    plt.axis([xmin, xmax, ymin, ymax])
    ax.set_zlim(-1.5, 1.5)
    plt.plot(x[abs(alph)>0.001],y[abs(alph)>0.001],'*g',markersize=10,markerfacecolor='g',zorder=100)
    
    # plt.contourf(X_plot,Y_plot,decision_val,zdir='z',offset=0)

    figure.canvas.draw(); figure.canvas.flush_events()
    if k<50-1:
        plt.cla()
        

    
print(alph)
print(beta)
print(alph+beta)
print(g)
print(h)
print(mu)
print(lam)
