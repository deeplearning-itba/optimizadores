import keras
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib.colors import ListedColormap
from keras import backend as K
import pickle
import os
import matplotlib.animation as animation
import pickle
from matplotlib import cm

def make_animation(output_filename,surface_filename,logWeights,max_frames=1000000):
    max_frames=np.min([len(logWeights.history["weights_0"]),max_frames])
    pesos_min=np.array(logWeights.history["weights_0"]).min()
    pesos_max=np.array(logWeights.history["weights_0"]).max()
    delta=np.abs(np.array(logWeights.history["weights_0"])[1:,0]-np.array(logWeights.history["weights_0"])[:-1,0])
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=720)
    with open(surface_filename, 'rb') as f:
        [w12,w21,w01,w10,w02,w20,J12,J01,J02]=pickle.load(f)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])        
    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(18, 12))
    ax1.contourf(w12, w21, J12, 50, cmap=cm, alpha=.8)
    ax2.contourf(w01, w10, J01, 50, cmap=cm, alpha=.8)
    ax4.contourf(w02, w20, J02, 50, cmap=cm, alpha=.8)  
    lr=np.array(logWeights.history["lr"])
    idx=np.arange(len(lr))
    decay=1/(1+(idx*logWeights.history["lr_decay"][0]))
    lr=lr*decay
    ax3.semilogy(lr,color="m")
    ax3_aux = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    ax5.semilogy(lr,color="m")
    ax5_aux = ax5.twinx() 
    ax1.scatter(logWeights.history["weights_0"][0][1],logWeights.history["weights_0"][0][2],c='y',label="start")
    ax1.scatter(logWeights.w1_opt,logWeights.w2_opt,marker='X',s=85,c='w',label="optimal")
    ax2.scatter(logWeights.history["weights_0"][0][0],logWeights.history["weights_0"][0][1],c='y',label="start")
    ax2.scatter(logWeights.w0_opt,logWeights.w1_opt,marker='X',s=85,c='w',label="optimal")
    ax4.scatter(logWeights.history["weights_0"][0][0],logWeights.history["weights_0"][0][2],c='y',label="start")
    ax4.scatter(logWeights.w0_opt,logWeights.w2_opt,marker='X',s=85,c='w',label="optimal")
    ax6.semilogy(lr,color="m")
    ax6.set_title("$\\Delta \\vec W $")
    ax6_aux = ax6.twinx() 
    ax1.set_title('w1 vs w2')
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')
    ax2.set_title('w0 vs w1')
    ax2.set_xlabel('w0')
    ax2.set_ylabel('w1')
    ax4.set_title('w0 vs w2')
    ax4.set_xlabel('w0')
    ax4.set_ylabel('w2')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Learning Rate',color="m")
    ax5.set_ylabel('Learning Rate',color="m")
    ax6.set_ylabel('Learning Rate',color="m")
    ax3_aux.set_xlabel('Iteración')
    ax3_aux.set_ylabel('Pesos')
    ax5_aux.set_xlabel('Epoch')
    ax5_aux.set_ylabel('Loss')
    ax6_aux.set_ylabel('Delta W')
    ax3_aux.legend()
    ax3_aux.set_ylim(pesos_min,pesos_max)
    ax6_aux.set_ylim(0,delta.max())
    ax5_aux.legend()
    ax1.legend()
    ax2.legend()
    ax4.legend()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    x=list(range(1000))
    loss=np.interp(x,logWeights.x,np.array(logWeights.losses))
    val_loss=np.interp(x,logWeights.x,np.array(logWeights.val_losses))
    loss_max=np.hstack([loss,val_loss]).max()
    ax5_aux.set_ylim(0,loss.max())
    ln1, = ax1.plot([],[],linewidth=5)
    ln2, = ax2.plot([],[],linewidth=5)
    ln4, = ax4.plot([],[],linewidth=5)
    ln3_1, = ax3_aux.plot([],[],linewidth=2)
    ln3_2, = ax3_aux.plot([],[],linewidth=2)
    ln3_3, = ax3_aux.plot([],[],linewidth=2)
    ln6, = ax6_aux.plot([],[],linewidth=2)
    ln5_1, = ax5_aux.plot([],[],linewidth=2)
    ln5_2, = ax5_aux.plot([],[],linewidth=2)
    sc1, = ax1.plot([],[],'go',c='k',label="end")
    sc2, = ax2.plot([],[],'go',c='k',label="end")
    sc4, = ax4.plot([],[],'go',c='k',label="end")
    def animate(i):
        print("\r{}".format(i),end="")
        ln1.set_data(np.array(logWeights.history["weights_0"])[:i,1],np.array(logWeights.history["weights_0"])[:i,2])
        ln2.set_data(np.array(logWeights.history["weights_0"])[:i,0],np.array(logWeights.history["weights_0"])[:i,1])
        ln4.set_data(np.array(logWeights.history["weights_0"])[:i,0],np.array(logWeights.history["weights_0"])[:i,2])
        ln5_1.set_data(x[:i],loss[:i])
        ln5_2.set_data(x[:i],val_loss[:i])
        ln6.set_data(x[:i],delta[:i])
        ln3_1.set_data(x[:i],np.array(logWeights.history["weights_0"])[:i,0])
        ln3_2.set_data(x[:i],np.array(logWeights.history["weights_0"])[:i,1])
        ln3_3.set_data(x[:i],np.array(logWeights.history["weights_0"])[:i,2])
        sc1.set_data(np.array(logWeights.history["weights_0"])[i,1],np.array(logWeights.history["weights_0"])[i,2])
        sc2.set_data(np.array(logWeights.history["weights_0"])[i,0],np.array(logWeights.history["weights_0"])[i,1])
        sc4.set_data(np.array(logWeights.history["weights_0"])[i,0],np.array(logWeights.history["weights_0"])[i,2])
        #     ax2.plot(np.array(logWeights.history["weights_0"])[:,0],np.array(logWeights.history["weights_0"])[:,1])
#     ax4.plot(np.array(logWeights.history["weights_0"])[:,0],np.array(logWeights.history["weights_0"])[:,2])
#     ax3_aux.plot(np.array(logWeights.history["weights_0"])[:,0],label='w0')
#     ax3_aux.plot(np.array(logWeights.history["weights_0"])[:,1],label='w1')
#     ax3_aux.plot(np.array(logWeights.history["weights_0"])[:,2],label='w2')
#     ax5_aux.plot(logWeights.x,np.array(logWeights.losses),label='train_loss')
#     ax5_aux.plot(logWeights.x,np.array(logWeights.val_losses),label='val_loss')
#     ax6_aux.plot(np.abs(np.array(logWeights.history["weights_0"])[1:,0]-np.array(logWeights.history["weights_0"])[:-1,0]))
    ani = animation.FuncAnimation(fig, animate, frames=max_frames, repeat=True,save_count=50)
    ani.save(output_filename, writer=writer)

class log_weights(keras.callbacks.Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """
    def __init__(self,get_weights,set_weights,surface_filename,w1_range,w2_range,w0_range,n_points,w1_opt,w2_opt,w0_opt,plotloss_data):
        super(log_weights, self).__init__()
        self.get_weights=get_weights   
        self.set_weights=set_weights
        self.weights=[]
        self.w1_range=w1_range
        self.w2_range=w2_range
        self.w0_range=w0_range
        self.n_points=n_points
        self.w1_opt=w1_opt
        self.w2_opt=w2_opt
        self.w0_opt=w0_opt
        self.plotloss_data=plotloss_data
        self.surface_filename=surface_filename
        
    def on_train_begin(self, logs=None):
        self.i=0
        self.x=list()
        self.epoch = []
        self.losses = []
        self.val_losses = []
        self.history = {}
        # Make data.
        if self.surface_filename:
            exists = os.path.isfile(self.surface_filename)
            if exists:
                with open(self.surface_filename, 'rb') as f:
                    [w12,w21,w01,w10,w02,w20,J12,J01,J02]=pickle.load(f)
            else:
                w1 = np.arange(self.w1_range[0], self.w1_range[1], (self.w1_range[1]-self.w1_range[0]) / self.n_points)
                w2 = np.arange(self.w2_range[0], self.w2_range[1], (self.w2_range[1]-self.w2_range[0]) / self.n_points)
                w0 = np.arange(self.w0_range[0], self.w0_range[1], (self.w0_range[1]-self.w0_range[0]) / self.n_points)
                w12,w21 = np.meshgrid(w1, w2)
                w01,w10 = np.meshgrid(w0, w1)
                w02,w20 = np.meshgrid(w0, w2)
                J12=np.zeros(w12.shape)
                J01=np.zeros(w01.shape)
                J02=np.zeros(w02.shape)
                for w1_i,w1_v in enumerate(w1):
                    for w2_i,w2_v in enumerate(w2):
                        J12[w1_i,w2_i]=self.get_loss(self.w0_opt,w12[w1_i,w2_i],w21[w1_i,w2_i])
                for w1_i,w1_v in enumerate(w0):
                    for w2_i,w2_v in enumerate(w1):
                        J01[w1_i,w2_i]=self.get_loss(w01[w1_i,w2_i],w10[w1_i,w2_i],self.w2_opt)
                for w1_i,w1_v in enumerate(w0):
                    for w2_i,w2_v in enumerate(w2):
                        J02[w1_i,w2_i]=self.get_loss(w02[w1_i,w2_i],self.w1_opt,w20[w1_i,w2_i])
                with open(self.surface_filename, 'wb') as f:
                    pickle.dump([w12,w21,w01,w10,w02,w20,J12,J01,J02],f)
        self.J12=J12
        self.J01=J01
        self.J02=J02
        self.w12=w12
        self.w21=w21
        self.w01=w01
        self.w10=w10
        self.w02=w02
        self.w20=w20
        try:
            logs["lr_decay"]=float(K.get_value(self.model.optimizer.decay))
        except:
            logs["lr_decay"]=0
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
#         self.on_epoch_end(0,logs)
        self.set_weights(self.model,0,0,0)
        self.on_batch_end(0,logs)
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.x.append(self.i)
            
        
    def on_batch_end(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        logs = logs or {}
        logs["lr"]=lr
        weights = self.get_weights(self.model)
        for idx,weight in enumerate(weights):
            logs['weights_{}'.format(idx)]=weight
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        #self.epoch.append(epoch)
        #self.weights.append(self.get_weights(self.model))
        self.i+=1
    
    def on_train_end(self,logs=None):
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        self.fig, [[self.ax1, self.ax2, self.ax3], [self.ax4, self.ax5, self.ax6]] = plt.subplots(2, 3, figsize=(18, 12))
        self.ax1.contourf(self.w12, self.w21, self.J12, 50, cmap=cm, alpha=.8)
        self.ax2.contourf(self.w01, self.w10, self.J01, 50, cmap=cm, alpha=.8)
        self.ax4.contourf(self.w02, self.w20, self.J02, 50, cmap=cm, alpha=.8)  
        self.ax1.plot(np.array(self.history["weights_0"])[:,1],np.array(self.history["weights_0"])[:,2])
        self.ax2.plot(np.array(self.history["weights_0"])[:,0],np.array(self.history["weights_0"])[:,1])
        self.ax4.plot(np.array(self.history["weights_0"])[:,0],np.array(self.history["weights_0"])[:,2])
        lr=np.array(self.history["lr"])
        idx=np.arange(len(lr))
        decay=1/(1+(idx*self.history["lr_decay"][0]))
        lr=lr*decay
        self.ax3.semilogy(lr,color="m")
        ax3_aux = self.ax3.twinx()  # instantiate a second axes that shares the same x-axis
        ax3_aux.plot(np.array(self.history["weights_0"])[:,0],label='w0')
        ax3_aux.plot(np.array(self.history["weights_0"])[:,1],label='w1')
        ax3_aux.plot(np.array(self.history["weights_0"])[:,2],label='w2')
        self.ax5.semilogy(lr,color="m")
        ax5_aux = self.ax5.twinx() 
        ax5_aux.plot(self.x,np.array(self.losses),label='train_loss')
        ax5_aux.plot(self.x,np.array(self.val_losses),label='val_loss')
        self.ax1.scatter(self.history["weights_0"][0][1],self.history["weights_0"][0][2],c='y',label="start")
        self.ax1.scatter(self.w1_opt,self.w2_opt,marker='X',s=85,c='w',label="optimal")
        self.ax2.scatter(self.history["weights_0"][0][0],self.history["weights_0"][0][1],c='y',label="start")
        self.ax2.scatter(self.w0_opt,self.w1_opt,marker='X',s=85,c='w',label="optimal")
        self.ax4.scatter(self.history["weights_0"][0][0],self.history["weights_0"][0][2],c='y',label="start")
        self.ax4.scatter(self.w0_opt,self.w2_opt,marker='X',s=85,c='w',label="optimal")
        self.ax1.scatter(self.history["weights_0"][-1][1],self.history["weights_0"][-1][2],c='k',label="end")
        self.ax2.scatter(self.history["weights_0"][-1][0],self.history["weights_0"][-1][1],c='k',label="end")
        self.ax4.scatter(self.history["weights_0"][-1][0],self.history["weights_0"][-1][2],c='k',label="end")
        self.ax6.semilogy(lr,color="m")
        self.ax6.set_title("$\\Delta \\vec W $")
        ax6_aux = self.ax6.twinx() 
        ax6_aux.plot(np.abs(np.array(self.history["weights_0"])[1:,0]-np.array(self.history["weights_0"])[:-1,0]))
        self.ax1.set_title('w1 vs w2')
        self.ax1.set_xlabel('w1')
        self.ax1.set_ylabel('w2')
        self.ax2.set_title('w0 vs w1')
        self.ax2.set_xlabel('w0')
        self.ax2.set_ylabel('w1')
        self.ax4.set_title('w0 vs w2')
        self.ax4.set_xlabel('w0')
        self.ax4.set_ylabel('w2')
        self.ax3.set_xlabel('Iteración')
        self.ax3.set_ylabel('Learning Rate',color="m")
        self.ax5.set_ylabel('Learning Rate',color="m")
        self.ax6.set_ylabel('Learning Rate',color="m")
        ax3_aux.set_xlabel('Iteración')
        ax3_aux.set_ylabel('Pesos')
        ax5_aux.set_xlabel('Epoch')
        ax5_aux.set_ylabel('Loss')
        ax6_aux.set_ylabel('Delta W')
        ax3_aux.legend()
        ax5_aux.legend()
        self.ax1.legend()
        self.ax2.legend()
        self.ax4.legend()
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()

    def get_loss(self,w0,w1,w2):
        self.set_weights(self.model,w0,w1,w2)
#         print(self.validation_data[0])
        return self.model.evaluate(self.plotloss_data[0],self.plotloss_data[1],verbose=0)