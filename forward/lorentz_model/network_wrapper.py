"""
Wrapper functions for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch import pow, add, mul, div, sqrt, abs, square, conj
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from torchsummary import summary
from torch.optim import lr_scheduler
from torchviz import make_dot
from utils.plotting import plot_weights_3D, plotMSELossDistrib, \
    compare_spectra, compare_spectra_with_params, plot_complex, plot_debug

# Libs
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelmax

import warnings
warnings.filterwarnings('ignore')


class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The network architecture object
        self.flags = flags                                      # The flags containing the hyperparameters
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # Network training mode, create a new ckpt folder
            if flags.model_name is None:                    # Use custom name if possible, otherwise timestamp
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.loss = self.make_custom_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train()
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train()
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for tensorboard
        self.best_validation_loss = float('inf')    # Set the BVL to large number
        self.best_pretrain_loss = float('inf')

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_data=(8,))
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('There are %d trainable out of %d total parameters' %(pytorch_total_params, pytorch_total_params_train))
        return model

    def make_MSE_loss(self, logit=None, labels=None):
        """
        Create a tensor that represents the loss. This is consistent both at training time \
        and inference time for a backward model
        :param logit: The output of the network
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels, reduction='mean')          # The MSE Loss of the network
        MSE_loss *= 10000
        # MSE_loss *= 1000

        return MSE_loss

    def make_custom_loss(self, logit1=None, logit2=None, labels=None):

        if logit1 is None:
            return None
        loss1 = nn.functional.mse_loss(logit1.real.float(), labels[:, : ,0].real.float(), reduction='mean')
        loss2 = nn.functional.mse_loss(logit1.imag.float(), labels[:, :, 0].imag.float(), reduction='mean')
        loss3 = nn.functional.mse_loss(logit2.real.float(), labels[:, :, 1].real.float(), reduction='mean')
        loss4 = nn.functional.mse_loss(logit2.imag.float(), labels[:, :, 1].imag.float(), reduction='mean')
        custom_loss = loss1 + loss2 + loss3 + loss4

        # boundary_loss1 = torch.sum(F.relu(abs(logit1.real) - 1)).float()
        # boundary_loss2 = torch.sum(F.relu(abs(logit1.imag) - 1)).float()
        # boundary_loss3 = torch.sum(F.relu(abs(logit2.real) - 1)).float()
        # boundary_loss4 = torch.sum(F.relu(abs(logit2.imag) - 1)).float()
        # custom_loss = loss1 + loss2 + loss3 + loss4 + \
        #               boundary_loss1 + boundary_loss2 + boundary_loss3 + boundary_loss4
        # custom_loss *= 1000

        # loss1 = nn.functional.mse_loss(logit1.float(), square(abs(labels[:, :, 0])).float(), reduction='mean')
        # loss2 = nn.functional.mse_loss(logit2.float(), square(abs(labels[:, :, 1])).float(), reduction='mean')
        # loss1 = 0
        # custom_loss = loss1 + loss2*1000
        return custom_loss


    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed.
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'AdamW':
            op = torch.optim.AdamW(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'Adamax':
            op = torch.optim.Adamax(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SparseAdam':
            op = torch.optim.SparseAdam(self.model.parameters(), lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale, momentum=0.9, nesterov=True)
        elif self.flags.optim == 'LBFGS':
            op = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise Exception("Optimizer is not available at the moment.")
        return op

    def make_lr_scheduler(self):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        # return lr_scheduler.StepLR(optimizer=self.optm, step_size=50, gamma=0.75, last_epoch=-1)
        return lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
                                        factor=self.flags.lr_decay_rate,
                                          patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model.pt
        :return: None
        """
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model.pt
        :return:
        """
        self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model.pt'))

    def init_weights(self):

        for layer_name, child in self.model.named_children():
            for param in self.model.parameters():
                if ('_w0' in layer_name):
                    # torch.nn.init.uniform_(child.weight, a=0.0, b=0.3)
                    torch.nn.init.xavier_uniform_(child.weight)
                elif ('_wp' in layer_name):
                    # torch.nn.init.uniform_(child.weight, a=0.0, b=0.1)
                    torch.nn.init.xavier_uniform_(child.weight)
                elif ('_g' in layer_name):
                    # torch.nn.init.uniform_(child.weight, a=0.0, b=0.01)
                    torch.nn.init.xavier_uniform_(child.weight)
                else:
                    if ((type(child) == nn.Linear) | (type(child) == nn.Conv2d)):
                        torch.nn.init.xavier_uniform_(child.weight)
                        if child.bias:
                            child.bias.data.fill_(0.00)

    def evaluate(self, save_dir='eval/'):
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()                       # Evaluation mode

        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(self.saved_model))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(self.saved_model))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(self.saved_model))
        # Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(self.saved_model))  # For pure forward model, there is no Xpred

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt, open(Ypred_file, 'a') as fyp:
            # Loop through the eval data and evaluate
            with torch.no_grad():
                for ind, (geometry, spectra) in enumerate(self.test_loader):
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit,w0,wp,g = self.model(geometry)
                    np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fyt, spectra.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fyp, logit.cpu().data.numpy(), fmt='%.3f')
        return Ypred_file, Ytruth_file

    def record_weight(self, name='Weights', layer=None, batch=999, epoch=999):
        """
        Record the weights for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0:
            weights = layer.weight.cpu().data.numpy()   # Get the weights

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')
            # print(weights_layer.shape)

            # Reshape the weights into a square dimension for plotting, zero padding if necessary
            wmin = np.amin(np.asarray(weights.shape))
            wmax = np.amax(np.asarray(weights.shape))
            sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
            diff = np.zeros((1, int(sq**2 - (wmin * wmax))), dtype='float64')
            weights = weights.reshape((1, -1))
            weights = np.concatenate((weights, diff), axis=1)
            # f = plt.figure(figsize=(10, 5))
            # c = plt.imshow(weights.reshape((sq, sq)), cmap=plt.get_cmap('viridis'))
            # plt.colorbar(c, fraction=0.03)
            f = plot_weights_3D(weights.reshape((sq, sq)), sq)
            self.log.add_figure(tag='1_Weights_' + name + '_Layer'.format(1),
                                figure=f, global_step=epoch)

    def record_grad(self, name='Gradients', layer=None, batch=999, epoch=999):
        """
        Record the gradients for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0 and epoch > 0:
            gradients = layer.weight.grad.cpu().data.numpy()

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')
            # print(weights_layer.shape)

            # Reshape the weights into a square dimension for plotting, zero padding if necessary
            wmin = np.amin(np.asarray(gradients.shape))
            wmax = np.amax(np.asarray(gradients.shape))
            sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
            diff = np.zeros((1, int(sq ** 2 - (wmin * wmax))), dtype='float64')
            gradients = gradients.reshape((1, -1))
            gradients = np.concatenate((gradients, diff), axis=1)
            # f = plt.figure(figsize=(10, 5))
            # c = plt.imshow(weights.reshape((sq, sq)), cmap=plt.get_cmap('viridis'))
            # plt.colorbar(c, fraction=0.03)
            f = plot_weights_3D(gradients.reshape((sq, sq)), sq)
            self.log.add_figure(tag='1_Gradients_' + name + '_Layer'.format(1),
                                figure=f, global_step=epoch)

    def train(self):
        """
        The major training function. This starts the training using parameters given in the flags
        :return: None
        """
        print("Starting training process")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler()
        self.init_weights()

        # # Start a tensorboard session for logging loss and training images
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        url = tb.launch()
        print("TensorBoard started at %s" % url)

        for epoch in range(self.flags.train_step):
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):

                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU

                self.optm.zero_grad()                                   # Zero the gradient first
                pred_r, pred_t = self.model(geometry)            # Get the output
                loss = self.make_custom_loss(pred_r, pred_t, spectra)
                # print(abs(spectra[:,:,1]))
                # if j == 0 and epoch == 0:
                #     im = make_dot(loss, params=dict(self.model.named_parameters())).render("Model Graph",
                #                                                                            format="png",
                #                                                                            directory=self.ckpt_dir)
                # print(loss)
                loss.backward()

                # Clip gradients to help with training
                if self.flags.use_clip:
                    if self.flags.use_clip:
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.flags.grad_clip)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.grad_clip)

                if epoch % self.flags.record_step == 0:
                    self.record_grad(name='eps_w0', layer=self.model.eps_w0, batch=j, epoch=epoch)
                    self.record_grad(name='eps_g', layer=self.model.eps_g, batch=j, epoch=epoch)

                if epoch % self.flags.record_step == 0:
                    for b in range(10):
                        if j == b:
                            for k in [0]:
                            # for k in range(self.flags.num_plot_compare):

                                # f = plot_complex(logit1=pred_t[k, :].cpu().data.numpy(),
                                #                  tr1 = square(spectra[k,:,1].abs()).cpu().data.numpy(),
                                #                  logit2=spectra[k, :, 1].real.cpu().data.numpy(),
                                #                  tr2 = spectra[k, :, 1].imag.cpu().data.numpy(),
                                #                  xmin=self.flags.freq_low, xmax=self.flags.freq_high,
                                #                  num_points=self.flags.num_spec_points)
                                # self.log.add_figure(tag='Test ' + str(k) +') Sample Transmission Spectrum'.format(1),
                                #                     figure=f, global_step=epoch)

                                logit1 = square(abs(pred_t)).cpu().data.numpy()
                                tr1 = square(abs(spectra[:,:,1])).cpu().data.numpy()
                                # logit2 = pred_t.cpu().data.numpy()
                                # tr2 = spectra[:,:,1].cpu().data.numpy()

                                f = plot_debug(logit1=logit1[k, :],tr1 = tr1[k, :], logit2=None,tr2 = None,
                                                 model=self.model, index=k, xmin=self.flags.freq_low,
                                                    xmax=self.flags.freq_high, num_points=self.flags.num_spec_points,
                                               num_osc=self.flags.num_lorentz_osc, y_axis='Transmission')
                                self.log.add_figure(tag='Test ' + str(k) + ' Batch ' + str(b) +
                                                        ' Debug Optical Constants'.format(1),
                                                    figure=f, global_step=epoch)



                self.optm.step()                                        # Move one step the optimizer
                train_loss.append(np.copy(loss.cpu().data.numpy()))     # Aggregate the loss

                self.model.eval()
                pred_r, pred_t = self.model(geometry)  # Get the output
                loss = self.make_custom_loss(pred_r, pred_t, spectra)
                train_loss_eval_mode_list.append(np.copy(loss.cpu().data.numpy()))
                self.model.train()

            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)

            if epoch % self.flags.eval_step == 0:           # For eval steps, do the evaluations and tensor board
                # Record the training loss to tensorboard
                self.log.add_scalar('Loss/ Training Loss', train_avg_loss, epoch)
                self.log.add_scalar('Loss/ Batchnorm Training Loss', train_avg_eval_mode_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                test_loss2 = []
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        pred_r, pred_t = self.model(geometry)  # Get the output
                        loss = self.make_custom_loss(pred_r, pred_t, spectra)

                        test_loss.append(np.copy(loss.cpu().data.numpy()))           # Aggregate the loss

                        # if j == 0 and epoch > 10 and epoch % self.flags.record_step == 0:
                        #     # f2 = plotMSELossDistrib(test_loss)
                        #     f2 = plotMSELossDistrib(logit.cpu().data.numpy(), spectra[:, ].cpu().data.numpy())
                        #     self.log.add_figure(tag='0_Testing Loss Histogram'.format(1), figure=f2,
                        #                         global_step=epoch)

                # Record the testing loss to the tensorboard
                test_avg_loss = np.mean(test_loss)
                self.log.add_scalar('Loss/ Validation Loss', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_eval_mode_loss, test_avg_loss))

                # Model improving, save the model
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        return None

            # # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
            # # self.lr_scheduler.step()


            if epoch > 10:
                restart_lr = self.flags.lr * 1
                if self.flags.use_warm_restart:
                    if epoch % self.flags.lr_warm_restart == 0:
                        for param_group in self.optm.param_groups:
                            param_group['lr'] = restart_lr
                            print('Resetting learning rate to %.5f' % restart_lr)

        # print('Finished')
        self.log.close()




    def pre_train(self):
        """
        The major training function. This starts the training using parameters given in the flags
        :return: None
        """
        print("Pre-train on test spectrum")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler()
        self.init_weights()

        # # Start a tensorboard session for logging loss and training images
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        url = tb.launch()
        print("TensorBoard started at %s" % url)

        for epoch in range(self.flags.train_step):
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):

                spectra_select = 0
                if j==0:
                    if cuda:
                        geometry = geometry[0:5].cuda()                          # Put data onto GPU
                        spectra = spectra[0:5].cuda()                            # Put data onto GPU

                    self.optm.zero_grad()                                   # Zero the gradient first
                    pred_r, pred_t = self.model(geometry)            # Get the output
                    loss = self.make_custom_loss(pred_r, pred_t, spectra)
                    # print(abs(spectra[:,:,1]))
                    # if j == 0 and epoch == 0:
                    #     im = make_dot(loss, params=dict(self.model.named_parameters())).render("Model Graph",
                    #                                                                            format="png",
                    #                                                                            directory=self.ckpt_dir)
                    # print(loss)
                    loss.backward()

                    if epoch % self.flags.record_step == 0:
                        self.record_grad(name='eps_w0', layer=self.model.eps_w0, batch=j, epoch=epoch)
                        self.record_grad(name='eps_g', layer=self.model.eps_g, batch=j, epoch=epoch)

                    if epoch % self.flags.record_step == 0:

                        for k in [0]:

                            # f = plot_complex(logit1=pred_t[k, :].cpu().data.numpy(),
                            #                  tr1 = square(spectra[k,:,1].abs()).cpu().data.numpy(),
                            #                  logit2=spectra[k, :, 1].real.cpu().data.numpy(),
                            #                  tr2 = spectra[k, :, 1].imag.cpu().data.numpy(),
                            #                  xmin=self.flags.freq_low, xmax=self.flags.freq_high,
                            #                  num_points=self.flags.num_spec_points)
                            # self.log.add_figure(tag='Test ' + str(k) +') Sample Transmission Spectrum'.format(1),
                            #                     figure=f, global_step=epoch)

                            logit1 = square(abs(pred_t)).cpu().data.numpy()
                            tr1 = square(abs(spectra[:,:,1])).cpu().data.numpy()
                            # logit2 = pred_t.cpu().data.numpy()
                            # tr2 = spectra[:,:,1].cpu().data.numpy()

                            f = plot_debug(logit1=logit1[k,:],tr1 = tr1[k,:], logit2=None,tr2 = None,
                                             model=self.model, index=0, xmin=self.flags.freq_low,
                                                xmax=self.flags.freq_high, num_points=self.flags.num_spec_points,
                                           num_osc=self.flags.num_lorentz_osc, y_axis='Transmission')
                            self.log.add_figure(tag=' Debug Optical Constants'.format(1),
                                                figure=f, global_step=epoch)



                    self.optm.step()                                        # Move one step the optimizer
                    train_loss.append(np.copy(loss.cpu().data.numpy()))     # Aggregate the loss

                    self.model.eval()
                    pred_r, pred_t = self.model(geometry)  # Get the output

                    loss1 = nn.functional.mse_loss(pred_r[0].real.float(), spectra[0, :, 0].real.float(), reduction='mean')
                    loss2 = nn.functional.mse_loss(pred_r[0].imag.float(), spectra[0, :, 0].imag.float(), reduction='mean')
                    loss3 = nn.functional.mse_loss(pred_t[0].real.float(), spectra[0, :, 1].real.float(), reduction='mean')
                    loss4 = nn.functional.mse_loss(pred_t[0].imag.float(), spectra[0, :, 1].imag.float(), reduction='mean')
                    loss = loss1 + loss2 + loss3 + loss4

                    train_loss_eval_mode_list.append(np.copy(loss.cpu().data.numpy()))
                    self.model.train()




            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)

            if epoch % self.flags.eval_step == 0:           # For eval steps, do the evaluations and tensor board
                # Record the training loss to tensorboard
                self.log.add_scalar('Loss/ Training Loss', train_avg_loss, epoch)
                self.log.add_scalar('Loss/ Batchnorm Training Loss', train_avg_eval_mode_loss, epoch)

                print("This is Epoch %d, training loss %.5f" \
                      % (epoch, train_avg_eval_mode_loss))

                # Model improving, save the model
                if train_avg_eval_mode_loss < self.best_validation_loss:
                    self.best_validation_loss = train_avg_eval_mode_loss
                    self.save()
                    print("Saving the model...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        return None

            # # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
            # # self.lr_scheduler.step()


            if epoch > 10:
                restart_lr = self.flags.lr * 1
                if self.flags.use_warm_restart:
                    if epoch % self.flags.lr_warm_restart == 0:
                        for param_group in self.optm.param_groups:
                            param_group['lr'] = restart_lr
                            print('Resetting learning rate to %.5f' % restart_lr)

        # print('Finished')
        self.log.close()




