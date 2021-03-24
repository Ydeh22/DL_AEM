"""
This file serves as a training interface for training the network
"""
# Built in
import os
import utils.training_data_utils as tdu
import utils.flagreader as fr
from network_wrapper import Network
from network_model import LorentzDNN
from utils.logging import write_flags_and_BVE


def training_from_flag(flags):
    """
    Training interface. 1. Read in data
                        2. Initialize network
                        3. Train network
                        4. Record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    if flags.use_cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # # Import the data
    train_loader, test_loader = tdu.generate_torch_dataloader(geoboundary=flags.geoboundary,
                                                              data_dir=flags.data_dir,
                                                              batch_size=flags.batch_size,
                                                              normalize_input=flags.normalize_input,
                                                              test_ratio=flags.test_ratio)

    # Reset the boundary if normalized
    if flags.normalize_input:
        flags.geoboundary_norm = [-1, 1, -1, 1]

    print("Geometry boundary is set to:", flags.geoboundary)

    # Make Network
    print("Making network now")
    ntwk = Network(LorentzDNN, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags object
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)
    # put_param_into_folder(ntwk.ckpt_dir)



if __name__ == '__main__':
    # Read the parameters to be set
    flags = fr.read_flag()

    # Call the train from flag function
    training_from_flag(flags)



