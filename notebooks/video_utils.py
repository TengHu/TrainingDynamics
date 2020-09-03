
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.offsetbox as offsetbox
from notebooks.analysis_utils import *

def save_video(name, num_iters):
    import subprocess #import call, check_call, PIPE, CalledProcessError, Popen
    #os.chdir("./videos")


    cmd_args = [
        'ffmpeg', '-y', '-framerate', str(num_iters),'-r', '6', '-i', './videos/file%02d.png', '-pix_fmt', 'yuv420p', '-vf', 
        'pad=ceil(iw/2)*2:ceil(ih/2)*2', name
    ]

    pipes = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_out, std_err = pipes.communicate()

    if pipes.returncode != 0:
        # an error happened!
        err_msg = "%s. Code: %s" % (std_err.strip(), pipes.returncode)
        raise Exception(err_msg)

    elif len(std_err):
        print ("SUCCESS!")

        
def viz_cosine_norm(norms_, cosines_,  corrects_, fc_, frames=[0], highlights=[], save_frame=False, func=None):
    r"""
    If only want to look at a subset of all examples, have to recompute cosines.
    """
    mean_ = fc_.mean(1)
   
    for i in frames:
        clear_output(wait=True)

        def plot_curve(ax_, metric, name):
            for chunk in chunks:
                ax_.plot(metric[:, chunk].mean(1))  
            ax_.legend([i for i in list(range(len(chunks)))], prop={'size': 15})
            ax_.set_title("iteration : " + str(i) + ", " + name)
            ax_.plot([i, i], [-1,1],  'r--')
           
            

        def plot_corrects_and_incorrects(ax_, metric1, metric2, name):
            to_plot = np.array(list(zip(metric1[i, :], metric2[i, :])))

            
            green = to_plot[corrects_[i], :]
            if len(green) > 0:
                ax_.scatter(green[:,0], green[:,1], marker = 'o', color="green", s=5)

            red = to_plot[~corrects_[i], :]
            if len(red) > 0:
                ax_.scatter(red[:,0], red[:,1], marker = 'o', color="red", s=5)

            ax_.set_ylim([-1.0, 1.0])
            
            
            
            
            for image_idx in highlights:    
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(trainset[image_idx][0].squeeze(0), cmap=plt.cm.gray_r), to_plot[image_idx], bboxprops =dict(edgecolor='green' if corrects_[i][image_idx] else 'red'))
                ax_.add_artist(imagebox)
                
            # line for cosine 0
            _, xmax = ax_.get_xlim()
            ax_.plot([0,xmax],[0,0], 'b--', linewidth=1)

            # line for batch gradient
            # batch_grad_norm = np.linalg.norm(mean_[i], ord=2, axis=-1)
            #ax_.plot([batch_grad_norm,batch_grad_norm], [-1,1], 'b--', linewidth=1)

            
            # deviate from previous batch gradient
            '''lag = 1
            if i >= lag:
                deviate_from_previous_iteration = mean_[i].dot(mean_[i-lag]) / batch_grad_norm / np.linalg.norm(mean_[i-lag], ord=2, axis=-1)
                
                
                ax_.plot([0,batch_grad_norm], [deviate_from_previous_iteration,deviate_from_previous_iteration], 'b', linewidth=5)'''
            ax_.set_title("iteration : " + str(i) + ", " + name)


        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        plot_corrects_and_incorrects(ax, norms_, cosines_, "fc2")
        
        
        
        # np.einsum("ba, bc", normed_batch_grad, normed_batch_grad)
        # normed_batch_grad = mean_ / np.linalg.norm(mean_, ord=2, axis=-1)[:, np.newaxis]

        # side plot
        if func is not None:
            func(ax[1],i)
       
        # save image
        if save_frame:
            plt.savefig("./videos/file%02d.png" % i,bbox_inches='tight')
        plt.show()
        
def viz_orthogonal_parallel1(norms_, angles_, corrects_, fc_, frames=[0], highlights=[], xlim = [-2, 2], ylim = [-10, 10], save_frame=False):
    r"""
    If only want to look at a subset of all examples, have to recompute angles.
    """
    num_examples = fc_.shape[1]
    mask = [1 if i % 2 else -1 for i in range(num_examples)]
    
    mean_ = fc_.mean(1)


    #qt = (num_examples * np.arange(0, 1, 1/len(chunks)))
    #ccolors = [colors[int(i),:] for i in qt]
    

    for i in frames:
        clear_output(wait=True)
        
        fig, ax = plt.subplots(figsize=(20, 10))

        
        y = (norms_[i] * np.sin(angles_[i])).clip(*ylim) * np.array(mask)[:,np.newaxis]
        x = (norms_[i] * np.cos(angles_[i])).clip(*xlim)
        
        
        ax.scatter(x[corrects_[i]], y[corrects_[i]], marker = 'o', color="green", s=5)
        ax.scatter(x[~corrects_[i]], y[~corrects_[i]], marker = 'o', color="red", s=5)
        
        ax.axvline(x=np.linalg.norm(mean_[i,:], ord=2, axis=-1).mean(), linewidth=1, color='blue')
        
        ax.plot(xlim, [0,0],  'b--')
        ax.plot([0,0],ylim,   'b--')
        ax.set_title("iteration: " + str(i))
        
        

        '''cones = cone_set_cover(fc2.mean(0)[i])
        for cone in cones:
            ax.plot([0, x[cone[0]]],[0,y[cone[0]]], '--', color="black", linewidth=.2)'''



        '''for chunk, c in zip(chunks, ccolors):
            ax[1].plot(corrects[:, :, chunk].mean((0,2)), color=c)
        ax[1].legend(range(len(chunks)))
        ax[1].axvline(i)'''
        
        
        for image_idx in highlights:    
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(trainset[image_idx][0].squeeze(0), cmap=plt.cm.gray_r), [x[image_idx],y[image_idx]], bboxprops =dict(edgecolor='green' if corrects_[i][image_idx] else 'red'))
                ax.add_artist(imagebox)
        
        
        if save_frame:
            plt.savefig("./videos/file%02d.png" % i,bbox_inches='tight')
        plt.show()
        
        
def viz_orthogonal_parallel2(norms_, angles_, corrects_, fc_, frames=[0], highlights=[], xlim = [-2, 2], ylim = [-10, 10], save_frame=False):
    r"""
    If only want to look at a subset of all examples, have to recompute angles.
    """
    
    def plot_batch_grad(batch_grad_norm_, lag=0, color_='orange', linewidth_=2):
        if i >= lag:
            norm_ = np.linalg.norm(mean_[i-lag], ord=2, axis=-1)
            angle_ = np.arccos(mean_[i].dot(mean_[i-lag]) / batch_grad_norm_ / norm_)
            ax.plot([0, norm_ * np.sin(angle_)],[0,norm_ * np.cos(angle_)], color=color_, linewidth=linewidth_)
    
    
    num_examples = fc_.shape[1]
    mask = [1 if i % 2 else -1 for i in range(num_examples)]
    
    
    mean_ = fc_.mean(1)
    

    for i in frames:
        clear_output(wait=True)
        
        fig, ax = plt.subplots(figsize=(10, 12))

        
        x = (norms_[i] * np.sin(angles_[i])).clip(*xlim) * np.array(mask)[:,np.newaxis]
        y = (norms_[i] * np.cos(angles_[i])).clip(*ylim)
        
        
        ax.scatter(x[corrects_[i]], y[corrects_[i]], marker = 'o', color="green", s=5)
        ax.scatter(x[~corrects_[i]], y[~corrects_[i]], marker = 'o', color="red", s=5)
        
        
        # plot current batch grad
        batch_grad_norm = np.linalg.norm(mean_[i], ord=2, axis=-1)
        #ax.plot([0, 0],[0,batch_grad_norm], color=[1, 98/255, 0], linewidth=5)
        
        #plot_batch_grad(batch_grad_norm, lag=2, linewidth_ = 5, color_=[253/255, 127/255, 44/255])
        #plot_batch_grad(batch_grad_norm, lag=4, linewidth_ = 5, color_=[253/255, 167/255, 102/255])
        
        ax.plot(xlim, [0,0],  'b--', linewidth=.5)
        ax.plot([0,0],ylim,   'b--', linewidth=.5)
        ax.set_title("iteration: " + str(i))
        
        

        '''cones = cone_set_cover(fc2.mean(0)[i])
        for cone in cones:
            ax.plot([0, x[cone[0]]],[0,y[cone[0]]], '--', color="black", linewidth=.2)'''



        '''for chunk, c in zip(chunks, ccolors):
            ax[1].plot(corrects[:, :, chunk].mean((0,2)), color=c)
        ax[1].legend(range(len(chunks)))
        ax[1].axvline(i)'''
        
        
        for image_idx in highlights:    
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(trainset[image_idx][0].squeeze(0), cmap=plt.cm.gray_r), [x[image_idx],y[image_idx]], bboxprops =dict(edgecolor='green' if corrects_[i][image_idx] else 'red'))
                ax.add_artist(imagebox)
        
        
        if save_frame:
            plt.savefig("./videos/file%02d.png" % i,bbox_inches='tight')
        plt.show()