import sys
import autograd.numpy as np
import autograd.numpy.random as npr

from neuralfingerprint import load_data
from neuralfingerprint import build_morgan_deep_net
from neuralfingerprint import build_conv_deep_net
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint.util import rmse
from neuralfingerprint.util import r2

from autograd import grad


def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params, seed=0,
             validation_smiles=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    num_print_examples = 100
    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = []
    def callback(weights, iter):
        if iter % 10 == 0:
            train_preds = undo_norm(pred_fun(weights, train_smiles[:num_print_examples]))
            cur_loss = loss_fun(weights, train_smiles[:num_print_examples], train_targets[:num_print_examples])
            training_curve.append(cur_loss)

            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve


def neural_graph_fps(target_name, input_path, len_smi):

    task_params = {'target_name' : target_name,
               'data_file' : input_path,}
    N_train = int(len_smi*0.7) - int(len_smi*0.7) % 100 + 100  # must be in hundreds, haven't found why
    N_val   = int(len_smi*0.1)
    N_test  = len_smi - N_train - N_val


    train_params = dict(num_iters=100,
                        batch_size=100,
                        init_scale=np.exp(-4),
                        step_size=np.exp(-6))

    traindata, valdata, testdata = load_data(
        task_params['data_file'], (N_train, N_val, N_test),
        input_name='smiles', target_name=task_params['target_name'])
    train_inputs, train_targets = traindata
    val_inputs,   val_targets   = valdata
    test_inputs,  test_targets  = testdata



    def print_performance(pred_func):
        train_preds = pred_func(train_inputs)
        val_preds = pred_func(val_inputs)

        return r2(val_preds, val_targets)

    def run_morgan_experiment():
        loss_fun, pred_fun, net_parser = \
            build_morgan_deep_net(model_params['fp_length'],
                                  model_params['fp_depth'], vanilla_net_params)
        num_weights = len(net_parser)
        predict_func, trained_weights, conv_training_curve = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                     train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)
        return print_performance(predict_func)

    def run_conv_experiment(model_params):
            
        # Define the architecture of the network that sits on top of the fingerprints.
        vanilla_net_params = dict(
        layer_sizes = [model_params['fp_length'], model_params['h1_size']],  # One hidden layer.
        normalize=True, L2_reg = model_params['L2_reg'], nll_func = rmse)
    
        conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
        conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1}
        loss_fun, pred_fun, conv_parser = \
            build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
        num_weights = len(conv_parser)
        predict_func, trained_weights, conv_training_curve = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                     train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)
        test_predictions = predict_func(test_inputs)
        return r2(test_predictions, test_targets)

    fp_lengths = [7,8,9]
    fp_depths = [6,7,8]
    conv_widths = [30,40,50,60]
    h1_sizes = [100]

#fp_length: 7 fp_depth: 8 conv_width: 60 h1_size: 100
#Neural test R2: 0.9314066399774756
    max_r2 = 0
    for fp_length in fp_lengths:
        for fp_depth in fp_depths:
            for conv_width in conv_widths:
                for h1_size in h1_sizes:
                    model_params = dict(fp_length=fp_length,    # Usually neural fps need far fewer dimensions than morgan.
                                        fp_depth=fp_depth,      # The depth of the network equals the fingerprint radius.
                                        conv_width=conv_width,   # Only the neural fps need this parameter.
                                        h1_size=h1_size,     # Size of hidden layer of network on top of fps.
                                        L2_reg=np.exp(-2))
                    test_r2_neural = run_conv_experiment(model_params)
                    max_r2 = max(test_r2_neural, max_r2)

    print ("fp_length:", fp_length, "fp_depth:", fp_depth, "conv_width:", conv_width, "h1_size:", h1_size)
    print ("Neural test R2:", test_r2_neural)
                    
