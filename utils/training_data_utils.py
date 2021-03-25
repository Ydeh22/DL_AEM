import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

def hdf5_to_ascii(data_dir, out_dir, suffix='', batch_size=None, S_params=False, spectra=True):

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
                # freq = h5['Spectra']['Freq'][:]
                labels_T = h5['Spectra']['Transmittance'][:]
                labels_R = h5['Spectra']['Reflectance'][:]
                labels_A = h5['Spectra']['Absorptance'][:]
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
                d_T = h5['Spectra']['Transmittance'][:]
                d_R = h5['Spectra']['Reflectance'][:]
                d_A = h5['Spectra']['Absorptance'][:]
                labels_T = np.vstack((labels_T, d_T))
                labels_R = np.vstack((labels_R, d_R))
                labels_A = np.vstack((labels_A, d_A))
            h5.close()
        x = (ind+1) - (current_batch-1)*batch_size
        if (x == batch_size) or (ind+1) == num_files:
            if ind == 0:
                break
            else:
                np.savetxt(os.path.join(out_dir, 'inputs_' + suffix
                                        + '_' + str(current_batch) + '.csv'), inputs, delimiter=',')
                if S_params:
                    np.savetxt(os.path.join(out_dir, 'S11(Re)_' + suffix
                                            + '_' + str(current_batch) + '.csv'), S11_re, delimiter=',')
                    np.savetxt(os.path.join(out_dir, 'S11(Im)_' + suffix
                                            + '_' + str(current_batch) + '.csv'), S11_im, delimiter=',')
                    np.savetxt(os.path.join(out_dir, 'S21(Re)_' + suffix
                                            + '_' + str(current_batch) + '.csv'), S21_re, delimiter=',')
                    np.savetxt(os.path.join(out_dir, 'S21(Im)_' + suffix
                                            + '_' + str(current_batch) + '.csv'), S21_im, delimiter=',')

                if spectra:
                    np.savetxt(os.path.join(out_dir, 'Trans_' + suffix
                                            + '_' + str(current_batch) + '.csv'), labels_T, delimiter=',')
                    np.savetxt(os.path.join(out_dir, 'Refl_' + suffix
                                            + '_' + str(current_batch) + '.csv'), labels_R, delimiter=',')
                    np.savetxt(os.path.join(out_dir, 'Abs_' + suffix
                                            + '_' + str(current_batch) + '.csv'), labels_A, delimiter=',')
                new_batch = 1

def check_data_distribution(data_dir):
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

def generate_torch_dataloader(geoboundary, normalize_input=True,
                              data_dir=os.path.abspath(''), batch_size=128,
                              rand_seed=1234, test_ratio = 0.2, shuffle = True):
    """

      :param batch_size: size of the batch read every time
      :param shuffle_size: size of the batch when shuffle the dataset
      :param data_dir: parent directory of where the data is stored, by default it's the current directory
      :param rand_seed: random seed
      :param test_ratio: if this is not 0, then split test data from training data at this ratio
                         if this is 0, use the dataIn/eval files to make the test set
      """

    # Import data files
    print('Importing data files...')
    geom = importData(os.path.join(data_dir, 'training_data'), 'inputs')
    s11_re = importData(os.path.join(data_dir, 'training_data'), 'S11(Re)')
    s11_im = importData(os.path.join(data_dir, 'training_data'), 'S11(Im)')
    s21_re = importData(os.path.join(data_dir, 'training_data'), 'S21(Re)')
    s21_im = importData(os.path.join(data_dir, 'training_data'), 'S21(Im)')
    s11 = np.expand_dims(s11_re + 1j * s11_im, axis=2)
    s21 = np.expand_dims(s21_re + 1j * s21_im, axis=2)
    scat = np.concatenate((s11,s21),axis=2)

    if (test_ratio > 0):
        print("Splitting data into training and test sets with a ratio of:", str(test_ratio))
        geom_Tr, geom_Te, scat_Tr, scat_Te = train_test_split(geom, scat,
                                                                test_size=test_ratio, random_state=rand_seed)
        print('Total number of training samples is {}'.format(len(geom_Tr)))
        print('Total number of test samples is {}'.format(len(geom_Te)))
        print('Length of an output spectrum is {}'.format(len(scat_Tr[1])))
    else:
        print("Using separate file from dataIn/Eval as test set")
        geom_Te, scat_Te = importData(os.path.join(data_dir, 'training_data', 'eval'))

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
    train_loader = FastTensorDataLoader(torch.from_numpy(geom_Tr),
                                        torch.from_numpy(scat_Tr), batch_size=batch_size, shuffle=shuffle)
    test_loader = FastTensorDataLoader(torch.from_numpy(geom_Te),
                                       torch.from_numpy(scat_Te), batch_size=batch_size, shuffle=shuffle)
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