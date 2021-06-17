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
from utils.data_processing_tools.process_CST_data import get_model_input_parameters, \
    get_spectra, get_extracted_opt_const

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

def lorentzian_osc_from_model(model_name, w_numpy, input, save_dir='Test',
                              save_name='Test', save_spectra=True, save_params=True, save_osc=True):

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

    # fig3,ax1 = plt.subplots(1,1)
    # ax1.plot(w_numpy, e1_eff,w_numpy, e2_eff,w_numpy, mu1_eff,w_numpy, mu2_eff)
    # fig3.show()

    if save_spectra:
        if os.path.isdir(save_dir) is False:
            os.mkdir(save_dir)

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
        if os.path.isdir(os.path.join(save_dir, 'Spectra')) is False:
            os.mkdir(os.path.join(save_dir, 'Spectra'))
        save_name_spectra = os.path.join(save_dir, 'Spectra', save_name + '_Spectra.txt')
        np.savetxt(save_name_spectra, np.transpose(save_spectra_data), fmt='%10.7f', header=header_spectra, delimiter='\t')

    if save_params:
        if os.path.isdir(save_dir) is False:
            os.mkdir(save_dir)
        save_param_data = np.empty((9, num_osc))
        save_param_data[0] = np.arange(1,num_osc+1)
        save_param_data[1] = (1 + abs(eps_params[3][0])).cpu().data.numpy()
        for i in range(3):
            save_param_data[2+i] = abs(eps_params[i][0]).cpu().data.numpy()
        save_param_data[5] = (1 + abs(mu_params[3][0])).cpu().data.numpy()
        for i in range(3):
            save_param_data[6+i] = abs(mu_params[i][0]).cpu().data.numpy()

        header_params = 'osc#\teps_inf\teps_w0\teps_wp\teps_g\tmu_inf\tmu_w0\tmu_wp\tmu_g'
        if os.path.isdir(os.path.join(save_dir, 'Params')) is False:
            os.mkdir(os.path.join(save_dir, 'Params'))
        save_name_params = os.path.join(save_dir, 'Params', save_name + '_Params.txt')
        np.savetxt(save_name_params, np.transpose(save_param_data), fmt='%10.7f', header=header_params, delimiter='\t')

    if save_osc:
        if os.path.isdir(save_dir) is False:
            os.mkdir(save_dir)
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

        if os.path.isdir(os.path.join(save_dir, 'Oscillators')) is False:
            os.mkdir(os.path.join(save_dir, 'Oscillators'))
        save_optConst = os.path.join(save_dir, 'Oscillators', save_name+'_OptConst.txt')
        np.savetxt(save_optConst, np.transpose(save_osc_data), fmt='%10.7f', header=header_optConst, delimiter='\t')

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
    models_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model/models'
    # models_dir = 'C:/Users/labuser/DL_AEM/forward/lorentz_model/models/'
    # models_dir = '/home/omar/PycharmProjects/DL_AEM/forward/lorentz_model/models'
    model_dir = os.path.join(models_dir, model_name)
    pytorch_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model'
    # pytorch_dir = 'C:/Users/labuser/DL_AEM/forward/lorentz_model/'
    # pytorch_dir = '/home/omar/PycharmProjects/DL_AEM/forward/lorentz_model'

    os.chdir(model_dir)
    flags = fr.read_flag()
    flags.eval_model = model_name

    ntwk = Network(LorentzDNN, flags, [], [], inference_mode=True, saved_model=model_name)
    ntwk.load()
    geoboundary = flags.geoboundary
    return ntwk.model, geoboundary

def lor_parameter_sweep_analysis(sweep_index, model_name, data_dir, save_name='Param_sweep'):

    models_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model/models'
    param_dir = os.path.join(models_dir,model_name,data_dir,'Params')
    # param_dir = os.path.join(data_dir, 'Params')
    filelist = os.listdir(param_dir)
    sweep_param = []
    for f in filelist:
        params = f.split('[')[1].split(']')[0].split(' ')[:-1]
        sweep_param.append(params[sweep_index])

    sweep_param = np.array(sweep_param)
    data_0 = np.loadtxt(os.path.join(param_dir, filelist[0]), delimiter='\t')

    num_params = sweep_param.shape[0]
    num_osc = data_0.shape[0]
    sweep_data = np.empty((2*(num_osc*3)+3,num_params))
    sweep_data[0] = sweep_param

    header = 'sweep_param\t'
    for p in ['eps_', 'mu_']:
        header += p+'inf\t'
        for q in ['w0_','wp_','g_']:
            for k in range(num_osc):
                header+=(p+q+str(k)+'\t')

    for ind,f in enumerate(filelist):
        data_temp = np.loadtxt(os.path.join(param_dir, f), delimiter='\t')
        sweep_data[1, ind] = data_temp[0, 1]
        sweep_data[int(2+3*num_osc), ind] = data_temp[0, 5]
        for j in [[2,2],[6,3+3*num_osc]]:
            for i in range(3):
                for k in range(num_osc):
                    x_index = int(j[1]+i*num_osc+k)
                    sweep_data[x_index,ind] = data_temp[k,j[0]+i]

    data = np.transpose(sweep_data)
    save_name = save_name+'.txt'
    save_file = os.path.join(models_dir,model_name,data_dir,save_name)
    np.savetxt(save_file,data,header=header,delimiter='\t')

def compare_cst_sim(data_dir, save_dir, sim_name, model_inputs, w):

    sim_folder = os.path.join(data_dir, sim_name)
    save_folder = os.path.join(save_dir, sim_name)
    if os.path.isdir(save_folder) is False:
        os.mkdir(save_folder)
    inputs,input_values = get_model_input_parameters(sim_folder, model_inputs)
    freq, spec = get_spectra(sim_folder)
    f2, eps, mu = get_extracted_opt_const(os.path.join(sim_folder, 'Extract Material Properties'))
    data = np.empty((8, freq[1].shape[0]))
    header = 'Freq\tR\tT\tA\te1_eff\te2_eff\tmu1_eff\tmu2_eff'
    data[0] = freq[1]
    data[1] = spec[1][1]
    data[2] = spec[0][1]
    data[3] = spec[2][1]
    data[4] = eps[0]
    data[5] = eps[1]
    data[6] = mu[0]
    data[7] = mu[1]
    np.savetxt(os.path.join(save_folder,sim_name+'.txt'), np.transpose(data), header=header, delimiter='\t')
    geom = np.array(input_values, dtype=np.float)
    lorentzian_osc_from_model(model_name, w, geom, save_dir=save_folder, save_name=sim_name+'_DNN',
                                  save_spectra=True, save_params=False, save_osc=False)

if __name__=='__main__':

    # model_name = '20210613_111143'
    # model_name = '20210614_142449_MSE2.9e-3'
    # model_name = '20210615_135030'
    model_name = '20210615_162928'

    models_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model/models'
    # models_dir = 'C:/Users/labuser/DL_AEM/forward/lorentz_model/models/'
    # models_dir = '/home/omar/PycharmProjects/DL_AEM/forward/lorentz_model/models'
    model_dir = os.path.join(models_dir, model_name)
    pytorch_dir = 'C:/Users/Omar/PycharmProjects/DL_AEM/forward/lorentz_model'

    # evaluate_from_model(model_name)

    cst_folder = 'C:/Users/Omar/OneDrive - Duke University/Padilla Group/DL Datasets/OK_Thermal_CST/ADM_SingleCyl_Thermal_IR/Export'

    freq_low = 20.02
    freq_high = 40
    num_points = 500
    w = generate_freq_axis(freq_low, freq_high, num_points)



    # geom = [r, h, p, loss]
    # r_min = 1.3,   r_max = 2.4
    # h_min = 0.975, h_max = 3
    # p_min = 6, p_max = 7
    # loss_min = 40, loss_max = 44

    sim_name = '0603-5839120'
    model_inputs = ['r', 'h', 'p', 'log_n_Si']

    compare_cst_sim(cst_folder,model_dir,sim_name,model_inputs,w)

    # r = 1.5783
    # h = 1.855875
    # p = 6.006
    # log_n_Si = 42.222502850232




    # geom = np.array([r, h, p, log_n_Si])

    # lorentzian_osc_from_model(model_name, w, geom, save_dir='CST_Comp', save_name='Test',
    #                               save_spectra=True, save_params=False, save_osc=False)

    # i = 10
    # geom = np.array([2, 1+i*0.2, 6.5, 40])
    # lorentzian_osc_from_model(model_name, w, geom, save_name='h_sweep_r=2_' + str(i),
    #                           save_spectra=True, save_params=False, save_osc=True)

    # geom = np.array([2, 1, 6.5, 40])
    # sweep_num = 10

    # for i in range(sweep_num):
    #     geom[1] = 1.0+i*(0.2)
    #     geometry = np.round(geom,3).astype(str)
    #
    #     geo_str = '['
    #     for i in range(geometry.shape[0]):
    #         geo_str += str(geometry[i])+' '
    #     geo_str += ']'
    #
    #
    #     lorentzian_osc_from_model(model_name, w, geom, save_dir='h_Sweep2', save_name='h_S_'+geo_str,
    #                               save_spectra=True, save_params=True, save_osc=True)


    # sweep_name = 'h_Sweep2'
    # lor_parameter_sweep_analysis(1, model_name, sweep_name, save_name=sweep_name+'_params')