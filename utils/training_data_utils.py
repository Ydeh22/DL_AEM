import os
import h5py
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

def hdf5_to_ascii(data_dir, out_dir, suffix='', batch_size=None, existing_batches=0, S_params=False, spectra=True, opt_const=False):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_list = []
    for file in os.listdir(os.path.join(data_dir)):
        if file.endswith('.h5'):
            file_list.append(file)
    num_files = len(file_list)
    if batch_size is None:
        batch_size = num_files
    new_batch = 1
    current_batch = 0
    for ind, f in enumerate(file_list):
        if new_batch == 1:
            new_batch = 0
            current_batch += 1
            h5 = h5py.File(os.path.join(data_dir,f),mode='r')
            g = h5['Model Variables']['Sweep Parameters']
            # params = g[0].astype('U13')
            inputs = g[1].astype(np.float)
            if S_params:
                # freq = h5['S-Parameters']['Freq'][:]
                S11_re = h5['S-Parameters']['S11 (Re)'][:][0]  # Just the fundamental mode
                S11_im = h5['S-Parameters']['S11 (Im)'][:][0]
                S21_re = h5['S-Parameters']['S21 (Re)'][:][0]
                S21_im = h5['S-Parameters']['S21 (Im)'][:][0]
            if spectra:
                freq_0 = h5['Spectra']['Freq'][:]
                labels_T = h5['Spectra']['Transmittance'][:]
                labels_R = h5['Spectra']['Reflectance'][:]
                labels_A = h5['Spectra']['Absorptance'][:]
                # labels_T = h5['Spectra']['Transmission'][:]
                # labels_R = h5['Spectra']['Reflection'][:]
                # labels_A = h5['Spectra']['Absorption'][:]
            if opt_const:
                # freq = h5['S-Parameters']['Freq'][:]
                e1 = h5['Optical Constants']['e1'][:]
                e2 = h5['Optical Constants']['e2'][:]
                mu1 = h5['Optical Constants']['mu1'][:]
                mu2 = h5['Optical Constants']['mu2'][:]

                e1_av = h5['Optical Constants']['e1_av'][:]
                e2_av = h5['Optical Constants']['e2_av'][:]
                mu1_av = h5['Optical Constants']['mu1_av'][:]
                mu2_av = h5['Optical Constants']['mu2_av'][:]


            h5.close()

        else:
            h5 = h5py.File(os.path.join(data_dir, f), mode='r')
            g = h5['Model Variables']['Sweep Parameters']
            input_x = g[1].astype(np.float)
            inputs = np.vstack((inputs, input_x))
            if S_params:
                d_S11_re = h5['S-Parameters']['S11 (Re)'][:][0]  # Just the fundamental mode
                d_S11_im = h5['S-Parameters']['S11 (Im)'][:][0]
                d_S21_re = h5['S-Parameters']['S21 (Re)'][:][0]
                d_S21_im = h5['S-Parameters']['S21 (Im)'][:][0]
                S11_re = np.vstack((S11_re, d_S11_re))
                S11_im = np.vstack((S11_im, d_S11_im))
                S21_re = np.vstack((S21_re, d_S21_re))
                S21_im = np.vstack((S21_im, d_S21_im))
            if spectra:
                freq = h5['Spectra']['Freq'][:]
                d_T = h5['Spectra']['Transmittance'][:]
                d_R = h5['Spectra']['Reflectance'][:]
                d_A = h5['Spectra']['Absorptance'][:]
                d_T = np.interp(freq_0, freq, d_T)
                d_R = np.interp(freq_0, freq, d_R)
                d_A = np.interp(freq_0, freq, d_A)
                # d_T = h5['Spectra']['Transmission'][:]
                # d_R = h5['Spectra']['Reflection'][:]
                # d_A = h5['Spectra']['Absorption'][:]
                labels_T = np.vstack((labels_T, d_T))
                labels_R = np.vstack((labels_R, d_R))
                labels_A = np.vstack((labels_A, d_A))
            if opt_const:
                d_e1 = h5['Optical Constants']['e1'][:]
                d_e2 = h5['Optical Constants']['e2'][:]
                d_mu1 = h5['Optical Constants']['mu1'][:]
                d_mu2 = h5['Optical Constants']['mu2'][:]
                e1 = np.vstack((e1, d_e1))
                e2 = np.vstack((e2, d_e2))
                mu1 = np.vstack((mu1, d_mu1))
                mu2 = np.vstack((mu2, d_mu2))

                d_e1_av = h5['Optical Constants']['e1_av'][:]
                d_e2_av = h5['Optical Constants']['e2_av'][:]
                d_mu1_av = h5['Optical Constants']['mu1_av'][:]
                d_mu2_av = h5['Optical Constants']['mu2_av'][:]
                e1_av = np.vstack((e1_av, d_e1_av))
                e2_av = np.vstack((e2_av, d_e2_av))
                mu1_av = np.vstack((mu1_av, d_mu1_av))
                mu2_av = np.vstack((mu2_av, d_mu2_av))

            h5.close()
        x = (ind+1) - (current_batch-1)*batch_size
        if (x == batch_size) or (ind+1) == num_files:
            if ind == 0:
                break
            else:
                np.savetxt(os.path.join(out_dir, 'inputs_' + suffix
                                        + '_' + str(current_batch+existing_batches) + '.csv'), inputs, delimiter=',')
                if S_params:
                    np.savetxt(os.path.join(out_dir, 'S11(Re)_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), S11_re, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'S11(Im)_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), S11_im, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'S21(Re)_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), S21_re, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'S21(Im)_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), S21_im, delimiter=',', fmt="%.7f")
                if spectra:
                    np.savetxt(os.path.join(out_dir, 'Trans_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), labels_T, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'Refl_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), labels_R, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'Abs_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), labels_A, delimiter=',', fmt="%.7f")
                if opt_const:
                    np.savetxt(os.path.join(out_dir, 'Eps(Re)_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), e1, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'Eps(Im)_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), e2, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'Mu(Re)_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), mu1, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'Mu(Im)_' + suffix
                                            + '_' + str(current_batch+existing_batches) + '.csv'), mu2, delimiter=',', fmt="%.7f")

                    np.savetxt(os.path.join(out_dir, 'Eps_Av(Re)_' + suffix
                                            + '_' + str(current_batch + existing_batches) + '.csv'), e1_av, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'Eps_Av(Im)_' + suffix
                                            + '_' + str(current_batch + existing_batches) + '.csv'), e2_av, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'Mu_Av(Re)_' + suffix
                                            + '_' + str(current_batch + existing_batches) + '.csv'), mu1_av, delimiter=',', fmt="%.7f")
                    np.savetxt(os.path.join(out_dir, 'Mu_Av(Im)_' + suffix
                                            + '_' + str(current_batch + existing_batches) + '.csv'), mu2_av, delimiter=',', fmt="%.7f")

                new_batch = 1

def check_data_distribution(data_dir):
    os.chdir(data_dir)
    train_data_files = []
    file_list = os.listdir(data_dir)
    data = None
    for ind, file in enumerate(file_list):
        if file.endswith('.csv'):
            if 'input' in file:
                # train_data_files.append(file)
                if data is None:
                    data = np.loadtxt(file,delimiter=',')
                else:
                    d = np.loadtxt(file, delimiter=',')
                    data = np.vstack((data, d))
    # print(data.shape[1])
    df = pd.DataFrame(data)
    hist = df.hist(bins=13, figsize=(10, 5))
    plt.tight_layout()
    plt.show()

def check_param_file_distribution(param_dir):
    train_data_files = []
    file_list = os.listdir(param_dir)
    data = None
    for ind, file in enumerate(file_list):
        if file.endswith('.txt'):
            if 'master' in file:
                # train_data_files.append(file)
                if data is None:
                    data = np.loadtxt(file, skiprows=1, delimiter='\t')
                else:
                    d = np.loadtxt(file, skiprows=1, delimiter='\t')
                    data = np.vstack((data, d))
    # print(data.shape[1])
    df = pd.DataFrame(data)
    hist = df.hist(bins=13, figsize=(10, 5))
    plt.tight_layout()
    plt.show()

def importData(data_dir, file_select):
    # Import raw data into python, should be either for training set or evaluation set
    train_data_files = []
    for file in os.listdir(os.path.join(data_dir)):
        if file.endswith('.csv'):
            if file_select in file:
                train_data_files.append(file)
    print(train_data_files)
    data = []
    for file_name in train_data_files:
        # Import full arrays
        data_array = pd.read_csv(os.path.join(data_dir, file_name), delimiter=',', header = None)
        data.extend(data_array.values)
    data = np.squeeze(np.array(data, dtype='float32'))
    return data

def generate_torch_dataloader(x_range, y_range, geoboundary, normalize_input=True,
                              data_dir=os.path.abspath(''), batch_size=128,
                              rand_seed=1234, test_ratio = 0.2, shuffle = True, dataset_size=0):
    """

      :param batch_size: size of the batch read every time
      :param shuffle_size: size of the batch when shuffle the dataset
      :param data_dir: parent directory of where the data is stored, by default it's the current directory
      :param rand_seed: random seed
      :param test_ratio: if this is not 0, then split test data from training data at this ratio
                         if this is 0, use the dataIn/eval files to make the test set
      """

    # Import data files
    # eval case
    if test_ratio == 0:
        folder_select = 'training_data/eval/'
        test_ratio = 0.999
        print('Importing data files from eval folder...')
    else:
        folder_select = 'training_data'
        print('Importing data files...')
    geom = importData(os.path.join(data_dir, folder_select), 'inputs')
    s11_re = importData(os.path.join(data_dir, folder_select), 'S11(Re)')
    s11_im = importData(os.path.join(data_dir, folder_select), 'S11(Im)')
    s21_re = importData(os.path.join(data_dir, folder_select), 'S21(Re)')
    s21_im = importData(os.path.join(data_dir, folder_select), 'S21(Im)')
    s11 = np.expand_dims(s11_re + 1j * s11_im, axis=2)
    s21 = np.expand_dims(s21_re - 1j * s21_im, axis=2)
    scat = np.concatenate((s11,s21),axis=2)

    if dataset_size != 0:
        data_reduce = dataset_size
    else:
        data_reduce = len(geom)

    geom = geom[:data_reduce]
    scat = scat[:data_reduce]

    # indices = y_range
    indices = []
    for i in range(1,len(geom)):
        # if geom[i,3] > 38:
        #     indices.append(i)
        # if geom[i,0] < 2.4:
        #     indices.append(i)
        if geom[i,3] > 37 and geom[i,0] < 2.0:
            indices.append(i)
        # if geom[i, 3] > 37:
        #     indices.append(i)

    if (test_ratio > 0):
        print("Splitting data into training and test sets with a ratio of:", str(test_ratio))

        geom_Tr, geom_Te, scat_Tr, scat_Te = train_test_split(geom[indices], scat[indices],
                                                                test_size=test_ratio, random_state=rand_seed)
        print('Total number of training samples is {}'.format(len(geom_Tr)))
        print('Total number of test samples is {}'.format(len(geom_Te)))
        print('Length of an output spectrum is {}'.format(len(scat_Tr[1])))
    else:
        print("Using separate file from dataIn/Eval as test set")
        geom_Te = importData(os.path.join(data_dir, 'training_data', 'eval'), 'inputs')
        s11_re = importData(os.path.join(data_dir, 'training_data', 'eval'), 'S11(Re)')
        s11_im = importData(os.path.join(data_dir, 'training_data', 'eval'), 'S11(Im)')
        s21_re = importData(os.path.join(data_dir, 'training_data', 'eval'), 'S21(Re)')
        s21_im = importData(os.path.join(data_dir, 'training_data', 'eval'), 'S21(Im)')
        s11 = np.expand_dims(s11_re + 1j * s11_im, axis=2)
        s21 = np.expand_dims(s21_re - 1j * s21_im, axis=2)
        scat_Te = np.concatenate((s11, s21), axis=2)
        geom_Tr = geom_Te
        scat_Tr = scat_Te

    print('Generating torch datasets')

    # Normalize the data if instructed using boundary
    if normalize_input:
        num_params = int(len(geoboundary)/2)
        for dset in [geom_Tr, geom_Te]:
            for p in range(num_params):
                dset[:, p::num_params] = \
                    (dset[:, p::num_params] - (geoboundary[p] +geoboundary[p+num_params]) / 2) / \
                    (geoboundary[p+num_params] - geoboundary[p]) * 2

    train_data = MetaMaterialDataSet(geom_Tr, scat_Tr, bool_train=True)
    test_data = MetaMaterialDataSet(geom_Te, scat_Te, bool_train=False)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    train_loader = FastTensorDataLoader(torch.from_numpy(geom_Tr[:, x_range]),
                                        torch.from_numpy(scat_Tr[:, y_range]), batch_size=batch_size, shuffle=shuffle)
    test_loader = FastTensorDataLoader(torch.from_numpy(geom_Te[:, x_range]),
                                       torch.from_numpy(scat_Te[:, y_range]), batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches