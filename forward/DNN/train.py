"""
This file serves as a training interface for training the network
"""
# Built in
import os
import utils.training_data_utils as tdu
import utils.flagreader as fr
from forward.DNN.network_wrapper import Network
from forward.DNN.network_model import Forward
from utils.logging import write_flags_and_BVE
from utils.plotting import plotMSELossDistrib_eval


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
    train_loader, test_loader = tdu.generate_torch_dataloader(x_range=flags.x_range,
                                                             y_range=flags.y_range,
                                                             geoboundary=flags.geoboundary,
                                                             batch_size=flags.batch_size,
                                                             normalize_input=flags.normalize_input,
                                                             data_dir=flags.data_dir,
                                                             test_ratio=flags.test_ratio,
                                                             dataset_size=flags.data_reduce)


    # Reset the boundary if normalized
    if flags.normalize_input:
        flags.geoboundary_norm = [-1, 1, -1, 1]

    print("Geometry boundary is set to:", flags.geoboundary)

    # Make Network
    print("Making network now")
    ntwk = Network(Forward, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags object
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.best_mse_loss, ntwk.ckpt_dir)
    # put_param_into_folder(ntwk.ckpt_dir)

def evaluate_from_model(model_dir):
    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :return: None
    """
    # Retrieve the flag object
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    print("Retrieving flag object for parameters")
    flags = fr.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode

    # Get the data
    # train_loader, test_loader = datareader.read_data(flags)
    train_loader, test_loader = tdu.generate_torch_dataloader(x_range=flags.x_range,
                                                             y_range=flags.y_range,
                                                             geoboundary=flags.geoboundary,
                                                             batch_size=flags.batch_size,
                                                             normalize_input=flags.normalize_input,
                                                             data_dir=flags.data_dir,
                                                             test_ratio=0,shuffle=False)

    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)

    # Evaluation process
    print("Start eval now:")
    pred_T_file, truth_T_file, pred_R_file, truth_R_file = ntwk.evaluate()

    # Plot the MSE distribution
    plotMSELossDistrib_eval(pred_T_file, truth_T_file, flags)
    print("Evaluation finished")


def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None


if __name__ == '__main__':
    # Read the parameters to be set
    # flags = fr.read_flag()
    #
    # # Call the train from flag function
    # training_from_flag(flags)

    # # Read the flag, however only the flags.eval_model is used and others are not used
    useless_flags = fr.read_flag()

    print(useless_flags.eval_model)
    # Call the evaluate function from model
    evaluate_from_model(useless_flags.eval_model)



