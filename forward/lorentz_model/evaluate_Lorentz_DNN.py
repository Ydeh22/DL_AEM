import os
import torch
import torch.nn.functional as F
from torch import pow, add, mul, div, sqrt, square
from torch import cos, sin, conj, abs, tan, log, exp, arctan
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd

from utils.plotting import plotMSELossDistrib_eval
from forward.lorentz_model.network_wrapper import Network
from forward.lorentz_model.network_model import LorentzDNN
from forward.lorentz_model.network_model import transfer_matrix
from utils.optical_constants import lorentzian
import utils.training_data_utils as tdu
import utils.flagreader as fr

def plot_example_fits(model_name, w):

    # pytorch_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model'
    pytorch_dir = 'C:/Users/labuser/DL_AEM/forward/lorentz_model/models/'
    eval_dir = os.path.join(pytorch_dir,'eval', model_name)
    os.chdir(eval_dir)

    # x_file = d+'test_Xtruth_'+model_name+'.csv'

    y1_file = eval_dir + '/test_Ytruth_' + model_name + '.csv'
    y2_file = eval_dir + '/test_Ypred_' + model_name + '.csv'

    df1 = pd.read_csv(y1_file, delimiter='\s', engine='python')
    df2 = pd.read_csv(y2_file, delimiter='\s', engine='python')

    y1 = df1.values
    y2 = df2.values

    index = 2

    data = np.zeros((3, y1[0].shape[0]))
    data[0] = w
    data[1] = df1.values[index - 1]
    data[2] = df2.values[index - 1]

    plt.plot(w, data[1], w, data[2])
    plt.show()
    # plt.ion()
    # plt.show(block=True)

    # mse = ((df1.values - df2.values)**2).mean(axis=1)
    mse = ((data[1] - data[2]) ** 2).mean(axis=0)
    print(mse)
    # print(data.shape)

    # os.chdir(model_dir)
    # np.savetxt('histogram_LorentzDNN.txt', mse,fmt='%10.7f',delimiter='\t')
    # np.savetxt('Curve'+str(index)+'_MSE_'+str(np.round(mse,6))+'.txt', np.transpose(data),fmt='%10.7f',delimiter='\t')

def lorentzian_osc_from_model(model_name, w_numpy, input, save_name='Test', save_spectra=True, save_params=True, save_osc=True):

    model, geoboundary = load_saved_model(model_name)
    num_params = int(len(geoboundary) / 2)

    cuda = True if torch.cuda.is_available() else False

    # geom = de_normalize(np.array([0.0, -0.0, 0.0, 0.0]),geoboundary)

    geom = input.copy()
    for p in range(num_params):
        geom[p::num_params] = \
            (geom[p::num_params] - (geoboundary[p] +geoboundary[p+num_params]) / 2) / \
            (geoboundary[p+num_params] - geoboundary[p]) * 2

    g_in = torch.tensor(geom, dtype=torch.float32).unsqueeze(dim=0)
    w = torch.tensor(w_numpy, dtype=torch.float32)
    if cuda:
        g_in = g_in.cuda()
        w = w.cuda()
    model.eval()
    with torch.no_grad():
        pred_r, pred_t = model(g_in)

    eps_params = model.eps_params_out
    mu_params = model.mu_params_out
    # print(eps_params)
    # print(mu_params)
    num_osc = eps_params[0][0].size()[0]

    h1, eps_re, eps_im, eps_inf = opt_const_from_param(eps_params, 'E', w)
    h2, mu_re, mu_im, mu_inf = opt_const_from_param(mu_params, 'M', w)

    e1 = np.sum(eps_re, axis=0) + eps_inf
    e2 = np.sum(eps_im, axis=0)
    mu1 = np.sum(mu_re, axis=0) + mu_inf
    mu2 = np.sum(mu_im, axis=0)
    e1_eff = model.eps_out.squeeze().real.cpu().data.numpy()
    e2_eff = model.eps_out.squeeze().imag.cpu().data.numpy()
    mu1_eff = model.mu_out.squeeze().real.cpu().data.numpy()
    mu2_eff = model.mu_out.squeeze().imag.cpu().data.numpy()


    fig3,ax1 = plt.subplots(1,1)
    ax1.plot(w_numpy, e1_eff,w_numpy, e2_eff,w_numpy, mu1_eff,w_numpy, mu2_eff)
    fig3.show()

    if save_spectra:
        save_spectra_data = np.empty((15,w_numpy.shape[0]))
        save_spectra_data[0] = w_numpy
        save_spectra_data[1] = square(abs(pred_r)).cpu().data.numpy()
        save_spectra_data[2] = square(abs(pred_t)).cpu().data.numpy()
        save_spectra_data[3] = pred_r.real.cpu().data.numpy()
        save_spectra_data[4] = pred_r.imag.cpu().data.numpy()
        save_spectra_data[5] = pred_t.real.cpu().data.numpy()
        save_spectra_data[6] = pred_t.imag.cpu().data.numpy()
        save_spectra_data[7] = e1
        save_spectra_data[8] = e2
        save_spectra_data[9] = mu1
        save_spectra_data[10] = mu2
        save_spectra_data[11] = e1_eff
        save_spectra_data[12] = e2_eff
        save_spectra_data[13] = mu1_eff
        save_spectra_data[14] = mu2_eff

        header_spectra = \
            'w\tR\tT\ts11(re)\ts11(im)\ts21(re)\ts21(im)\te1\te2\tmu1\tmu2\te1_eff\te2_eff\tmu1_eff\tmu2_eff'
        save_name_spectra = os.path.join(save_name + '_Spectra.txt')
        np.savetxt(save_name_spectra, np.transpose(save_spectra_data), fmt='%10.7f', header=header_spectra, delimiter='\t')

    if save_params:
        save_param_data = np.empty((8, num_osc))
        save_param_data[0] = (1 + abs(eps_params[3][0])).cpu().data.numpy()
        for i in range(3):
            save_param_data[1+i] = abs(eps_params[i][0]).cpu().data.numpy()
        save_param_data[4] = (1 + abs(mu_params[3][0])).cpu().data.numpy()
        for i in range(3):
            save_param_data[5+i] = abs(mu_params[i][0]).cpu().data.numpy()

        header_params = 'eps_inf\teps_w0\teps_wp\teps_g\tmu_inf\tmu_w0\tmu_wp\tmu_g'
        save_name_params = os.path.join(save_name + '_Params.txt')
        np.savetxt(save_name_params, np.transpose(save_param_data), fmt='%10.7f', header=header_params, delimiter='\t')

    if save_osc:
        save_osc_data = np.empty((int(7+4*num_osc),w_numpy.shape[0]))
        header_optConst = 'w\te1\te2\tmu1\tmu2\teps_inf\tmu_inf\t'+h1+h2
        save_osc_data[0] = w_numpy
        save_osc_data[1] = e1
        save_osc_data[2] = e2
        save_osc_data[3] = mu1
        save_osc_data[4] = mu2
        save_osc_data[5] = eps_inf
        save_osc_data[6] = mu_inf

        for i in range(num_osc):
            save_osc_data[7 + int(2 * i)] = eps_re[i]
            save_osc_data[7 + int(2 * i + 1)] = eps_im[i]

        for i in range(num_osc):
            save_osc_data[7 + int(2*num_osc) + int(2 * i)] = mu_re[i]
            save_osc_data[7 + int(2*num_osc) + int(2 * i + 1)] = mu_im[i]

        save_optConst = os.path.join(save_name+'_OptConst.txt')
        np.savetxt( save_optConst, np.transpose(save_osc_data), fmt='%10.7f', header=header_optConst, delimiter='\t')

    torch.cuda.empty_cache()


def opt_const_from_param(params, osc, w):

    w0 = abs(params[0][0])
    wp = abs(params[1][0])
    g = abs(params[2][0])
    p_inf = abs(params[3][0])

    num_osc = params[0][0].size()[0]
    re = np.empty((num_osc, w.size()[0]))
    im = np.empty_like(re)
    header = ''
    inf = 1+p_inf.expand_as(w).cpu().data.numpy()

    for i in range(num_osc):
        e1, e2 = lorentzian(w, w0[i], wp[i], g[i])
        header = header+osc+str(i+1)+'_re\t' + osc+str(i+1)+'_im\t'
        re[i] = e1.cpu().data.numpy()
        im[i] = e2.cpu().data.numpy()

    return header, re, im, inf


def evaluate_from_model(model_name):

    models_dir = 'C:/Users/Omar/OneDrive - Duke University/Padilla Group/Manuscripts/Lorentz DNN/DNN models/Lorentz/06-2021'
    model_dir = os.path.join(models_dir, model_name)
    pytorch_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model'

    os.chdir(pytorch_dir)
    flags = fr.read_flag()
    # flags = fr.load_flags(model_dir)
    flags.eval_model = model_name


    print("Load data:")
    train_loader, test_loader = tdu.generate_torch_dataloader(x_range=flags.x_range,
                                                              y_range=flags.y_range,
                                                              geoboundary=flags.geoboundary,
                                                              batch_size=flags.batch_size,
                                                              normalize_input=flags.normalize_input,
                                                              data_dir=flags.data_dir,
                                                              test_ratio=0.999, shuffle=False)

    ntwk = Network(LorentzDNN, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)

    # Evaluation process
    print("Start eval now:")
    pred_file, truth_file = ntwk.evaluate()

    # Plot the MSE distribution
    plotMSELossDistrib_eval(pred_file, truth_file, flags)
    print("Evaluation finished")


def generate_freq_axis(freq_low, freq_high, num_points):
    w = np.arange(freq_low, freq_high, (freq_high - freq_low) / num_points)
    return w

def de_normalize(params, geoboundary):
    num_params = len(params)
    for p in range(num_params):
        params[p] = params[p] * 0.5 * (geoboundary[p+num_params] - geoboundary[p]) + (
                    geoboundary[p+num_params] + geoboundary[p]) * 0.5
        return params

def load_saved_model(model_name):

    # models_dir = 'C:/Users/Omar/OneDrive - Duke University/Padilla Group/Manuscripts/Lorentz DNN/DNN models/Lorentz/06-2021'
    # models_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model/models'
    # models_dir = 'C:/Users/labuser/DL_AEM/forward/lorentz_model/models/'
    models_dir = '/home/omar/PycharmProjects/DL_AEM/forward/lorentz_model/models'
    model_dir = os.path.join(models_dir, model_name)
    # pytorch_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model'
    # pytorch_dir = 'C:/Users/labuser/DL_AEM/forward/lorentz_model/'
    pytorch_dir = '/home/omar/PycharmProjects/DL_AEM/forward/lorentz_model'

    os.chdir(model_dir)
    flags = fr.read_flag()
    flags.eval_model = model_name

    ntwk = Network(LorentzDNN, flags, [], [], inference_mode=True, saved_model=model_name)
    ntwk.load()
    geoboundary = flags.geoboundary
    return ntwk.model, geoboundary

if __name__=='__main__':

    # model_name = '20210613_111143'
    # model_name = '20210614_142449_MSE2.9e-3'
    # model_name = '20210615_135030'
    model_name = '20210615_162928'


    freq_low = 20.02
    freq_high = 40
    num_points = 500
    w = generate_freq_axis(freq_low, freq_high, num_points)

    # evaluate_from_model(model_name)

    # plot_example_fits(model_name, w)

    # geom = [r, h, p, loss]
    # r_min = 1.3,   r_max = 2.4
    # h_min = 0.975, h_max = 3
    # p_min = 6, p_max = 7
    # loss_min = 40, loss_max = 44

    geom = np.array([1.3, 2, 6.5, 40])

    # geom = np.array([2.0678, 1.4124, 6.715, 43.583330640191])

    lorentzian_osc_from_model(model_name, w, geom, save_name='0603-5642859_',
                              save_spectra=False, save_params=False, save_osc=False)

    # i = 10
    # geom = np.array([2, 1+i*0.2, 6.5, 40])
    # lorentzian_osc_from_model(model_name, w, geom, save_name='h_sweep_r=2_' + str(i),
    #                           save_spectra=True, save_params=False, save_osc=True)

    # for i in range(2):
    #     geom[0] = 1.3+i*(0.1)
    #
    #     # lorentzian_osc_from_model(model_name, w, geom, save_name='Test_'+str(i),
    #     #                           save_spectra=True, save_params=False, save_osc=False)
    #     lorentzian_osc_from_model(model_name, w, geom, save_name='0603-5642859_',
    #                               save_spectra=False, save_params=True, save_osc=False)



