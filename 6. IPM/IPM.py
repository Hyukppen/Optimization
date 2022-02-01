import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import time
# %% interior point method - cost function
plt.close('all')
x1=np.linspace(-1.2,1.2,100)
x2=np.linspace(-1.2,1.2,100)
X1,X2=np.meshgrid(x1,x2)

t=100
f=X1+X2
g=X1**2+X2**2-1
L = f - t*np.log(-g)

L[np.isnan(L)]=np.max(L[~np.isnan(L)])
L=L-np.min(L)
L=L/np.max(L)

figure = plt.figure(figsize=[6, 9])
ax = figure.gca(projection="3d")
# ax.set_box_aspect((1,1,2))
ls = LightSource(azdeg=-45, altdeg=10)
ax.view_init(elev=10,azim=-45)
rgb = ls.shade(L, plt.cm.RdYlBu)
ax.plot_surface(X1,X2,L,facecolors=rgb); plt.xlabel('x1'); plt.ylabel('x2')
# ax.plot_surface(X1,X2,L, cmap='plasma',facecolors=rgb)
plt.axis([min(x1), max(x1), min(x2), max(x2)])
ax.set_zlim(0, 1)
plt.tight_layout()
# %% interior point method - cost function (video)
plt.close('all')
x1=np.linspace(-1.2,1.2,15)
x2=np.linspace(-1.2,1.2,15)
X1,X2=np.meshgrid(x1,x2)

t=0.01
f=X1+X2
g=X1**2+X2**2-1

figure = plt.figure(figsize=[8, 9])
high=0;
low=1;
while(1):
    if t > 20:
        high=1
        low=0
    if t < 0.01:
        low=1
        high=0
    if high==1:
        t=0.7*t
    if low==1:
        t=1.3*t

    L = f - t*np.log(-g)
    L[np.isnan(L)]=np.max(L[~np.isnan(L)])
    L=L-np.min(L)
    L=L/np.max(L)

    ax = figure.gca(projection="3d")
    # ax.set_box_aspect((1,1,2))
    ls = LightSource(azdeg=-45, altdeg=10)
    ax.view_init(elev=10,azim=-45)
    rgb = ls.shade(L, plt.cm.RdYlBu)
    ax.plot_surface(X1,X2,L, facecolors=rgb); plt.xlabel('x1'); plt.ylabel('x2')
    # ax.plot_surface(X1,X2,L, cmap='plasma',facecolors=rgb)
    plt.axis([min(x1), max(x1), min(x2), max(x2)])
    ax.set_zlim(0, 1)
    plt.tight_layout()
    figure.canvas.draw(); figure.canvas.flush_events(); plt.cla();

# %% interior point method - IPM
# xk=np.array([[0],[0]])
# tl=1
# for l in range(50):
#     tl1=0.8*tl
#     for k in range(6):
#         x1=xk[0]
#         x2=xk[1]
#         t=tl1
#         g=np.vstack([1+t*(2*x1)/(-x1**2-x2**2+1) , 1+t*(2*x2)/(-x1**2-x2**2+1)])
#         H=np.block([[t*(2*x1**2-2*x2**2+2)/(1-x1**2-x2**2)**2, t*4*x1*x2/(1-x1**2-x2**2)**2], \
#                     [t*4*x1*x2/(1-x1**2-x2**2)**2, t*(-2*x1**2+2*x2**2+2)/(1-x1**2-x2**2)**2]])
        
#         xk1=xk-np.linalg.inv(H)@g
#         xk=xk1
#     tl=tl1
# %% interior point method - Primal-Dual IPM
# wk=np.array([ [0], [0], [1] ])
# tl=1
# for k in range(45):
#     tl1=0.8*tl
    
#     x1=wk[0]
#     x2=wk[1]
#     mu=wk[2]
    
#     t=tl1
#     R=np.vstack([1+2*x1*mu , 1+2*x2*mu , mu*(x1**2+x2**2-1)+t ])
#     dRdw=np.block([ [2,          0,         2*x1], \
#                     [0,          2,         2*x2], \
#                     [2*mu*x1, 2*mu*x2, x1**2+x2**2-1]])
    
#     wk1=wk-np.linalg.inv(dRdw)@R
#     wk=wk1
#     tl=tl1
# %% interior point method - IPM (for plot)
xk=np.array([[0],[0]])
tl=1
    
plt.close('all')
x1_plot=np.linspace(-1,1,100)
x2_plot=np.linspace(-1,1,100)
X1,X2=np.meshgrid(x1_plot,x2_plot)
f=X1+X2
g=X1**2+X2**2-1
t=tl
L = f - t*np.log(-g)
L[np.isnan(L)]=np.max(L[~np.isnan(L)])
figure=plt.figure(figsize=[9, 9.5])
plt.contour(x1_plot,x2_plot,L,30); plt.xlabel('x1'); plt.ylabel('x2'); plt.grid()
plt.axis('image'); plt.axis([-1.1, 0.4, -1.1, 0.4])
plt.plot(xk[0],xk[1],'ko',markersize=11)
xy_plot=xk
figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(.01)

for l in range(50):
    tl1=0.8*tl;
    
    L = f - tl1*np.log(-g)
    L[np.isnan(L)]=np.max(L[~np.isnan(L)])
    
    plt.cla()
    plt.contour(x1_plot,x2_plot,L,30); plt.xlabel('x1'); plt.ylabel('x2'); plt.grid()
    plt.axis('image'); plt.axis([-1.1, 0.4, -1.1, 0.4])
    figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(.01)
    for k in range(6):
        x1=xk[0]
        x2=xk[1]
        t=tl1
        grad=np.vstack([1+t*(2*x1)/(-x1**2-x2**2+1) , 1+t*(2*x2)/(-x1**2-x2**2+1)])
        H=np.block([[t*(2*x1**2-2*x2**2+2)/(1-x1**2-x2**2)**2, t*4*x1*x2/(1-x1**2-x2**2)**2], \
                    [t*4*x1*x2/(1-x1**2-x2**2)**2, t*(-2*x1**2+2*x2**2+2)/(1-x1**2-x2**2)**2]])
        
        xk1=xk-np.linalg.inv(H)@grad
        xk=xk1
        xy_plot=np.hstack([xy_plot, xk])
        plt.plot(xy_plot[0,:],xy_plot[1,:],'ko--',markersize=11)
        figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(.01)
    tl=tl1
    xy_plot=xk;
    plt.plot(xy_plot[0,:],xy_plot[1,:],'r*',markersize=15)
    figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(.2)


# %% interior point method - Primal-Dual IPM (for plot)
wk=np.array([ [0], [0], [1] ])
xk=wk[:2]
tl=1
    
plt.close('all')
x1_plot=np.linspace(-1,1,100)
x2_plot=np.linspace(-1,1,100)
X1,X2=np.meshgrid(x1_plot,x2_plot)
f=X1+X2
g=X1**2+X2**2-1
t=tl
L = f - t*np.log(-g)
L[np.isnan(L)]=np.max(L[~np.isnan(L)])
figure=plt.figure(figsize=[9, 9.5])
plt.contour(x1_plot,x2_plot,L,30); plt.xlabel('x1'); plt.ylabel('x2'); plt.grid()
plt.axis('image'); plt.axis([-1.1, 0.4, -1.1, 0.4])
plt.plot(xk[0],xk[1],'ko',markersize=11)
xy_plot=wk[0:2]
figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(1)

for k in range(45):
    tl1=0.8*tl
    
    x1=wk[0]
    x2=wk[1]
    mu=wk[2]
    
    t=tl1
    R=np.vstack([1+2*x1*mu , 1+2*x2*mu , mu*(x1**2+x2**2-1)+t ])
    dRdw=np.block([ [2,          0,         2*x1], \
                    [0,          2,         2*x2], \
                    [2*mu*x1, 2*mu*x2, x1**2+x2**2-1]])
    
    wk1=wk-np.linalg.inv(dRdw)@R
    wk=wk1
    tl=tl1
    
    L = f - tl1*np.log(-g)
    L[np.isnan(L)]=np.max(L[~np.isnan(L)])
    xy_plot=np.hstack([xy_plot, wk[:2]])
    plt.cla()
    plt.contour(x1_plot,x2_plot,L,30); plt.xlabel('x1'); plt.ylabel('x2'); plt.grid()
    plt.axis('image'); plt.axis([-1.1, 0.4, -1.1, 0.4])
    plt.plot(xy_plot[0,:],xy_plot[1,:],'ro--',markersize=11)
    figure.canvas.draw(); figure.canvas.flush_events(); time.sleep(.1)
