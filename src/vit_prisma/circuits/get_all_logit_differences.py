
import h5py
import os
import numpy as np
from scipy.special import logsumexp
import argparse
import json
import time

# Path to your HDF5 file

def read_accuracies_from_file(file_path):
    """
    Reads a .txt file and extracts accuracy values formatted like "Accuracy for Layer 0, Head 0: 0.7198".

    :param file_path: Path to the .txt file.
    :return: A dictionary with the layer and head as keys and the accuracy as values.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Accuracy for Layer' in line:
                    parts = line.strip().split(',')
                    parts = parts[-1].split(':')
                    acc = parts[-1].strip()
        return float(acc)
    except Exception as e:
        print(f"Error reading file: {e}")

def get_logits(ablation_type, parent_dir = '/network/scratch/s/sonia.joseph/imagenet_logits/'):
    load_path = os.path.join(parent_dir, ablation_type)
    with h5py.File(os.path.join(load_path, 'logits_and_labels.h5'), 'r') as h5f:
        logits = h5f['logits']
        labels = h5f['labels']
        return logits[:], labels[:]


def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    
    :param x: A numpy array or a list of numbers.
    :return: A numpy array of softmax values.
    """
    # Convert input to a numpy array if it's not already
    x = np.array(x)
    
    # Calculate the exponentials of each element and the sum of those exponentials
    exp_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    sum_exp_x = np.sum(exp_x)
    
    # Compute softmax
    return exp_x / sum_exp_x


def get_logit_diff(ablation_type, specific_label, vanilla_logits, save_dir, logit_dir = '/network/scratch/s/sonia.joseph/imagenet_logits/'):
    
    vanilla_logits, labels = get_logits('unablated')

    # Get original logits
    ablated_logits, _ = get_logits(ablation_type)

    # Get datapoints for chosen label
    indices_of_specific_label = np.where(labels[:] == specific_label)[0]
    specific_vanilla = vanilla_logits[indices_of_specific_label]
    specific_ablated = ablated_logits[indices_of_specific_label]
    
    # Raw logit value
    raw_vanilla_logit = specific_vanilla[:, specific_label].mean()
    raw_ablated_logit = specific_ablated[:, specific_label].mean()
    
    raw_vanilla_softmax = softmax(specific_vanilla)[:, specific_label].mean()
    raw_ablated_softmax = softmax(specific_ablated)[:, specific_label].mean()
    
    # Get absolute difference
    abs_diff = specific_vanilla - specific_ablated
    abs_diff = abs_diff[:,specific_label]
    abs_diff = abs_diff.mean()
    
    abs_percent = abs_diff / specific_vanilla[:,specific_label].mean()
    
    # Get relative logit difference (difference between logit differences)
    all_indices_except_specific = [i for i in range(1000) if i != specific_label]
    vanilla_diff = specific_vanilla[:,all_indices_except_specific].mean() - specific_vanilla[:, specific_label].mean()
    ablated_diff = specific_ablated[:,all_indices_except_specific].mean() - specific_ablated[:, specific_label].mean()
    
    #  Get relative difference
    rel_diff = vanilla_diff - ablated_diff
    rel_percent = rel_diff / vanilla_diff
    
    # Get logsumexp
    lse_vanilla = logsumexp(specific_vanilla[:,all_indices_except_specific] - logsumexp(specific_vanilla[:, specific_label]))
    lse_ablated = logsumexp(specific_ablated[:,all_indices_except_specific] - logsumexp(specific_ablated[:, specific_label]))
    
    # Get softmax difference
    softmax_diff = softmax(specific_vanilla)[:,specific_label].mean() - softmax(specific_ablated)[:,specific_label].mean()
    softmax_percent = softmax_diff / softmax(specific_vanilla)[:,specific_label].mean()    
    
    data = {
        "raw_vanilla_logit": raw_vanilla_logit,
        "raw_ablated_logit": raw_ablated_logit,
        "raw_vanilla_softmax": raw_vanilla_softmax,
        "raw_ablated_softmax": raw_ablated_softmax,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "-abs_percent": -abs_percent, # Change from vanilla
        "-rel_percent": -rel_percent, # Change from vanilla
        "lse_vanilla": lse_vanilla,
        "lse_ablated": lse_ablated,
        "softmax_diff": softmax_diff,
        "-softmax_percent": -softmax_percent # Change from vanilla 
    }

    return data

def numpy_to_native(data):
    """Converts a structure (possibly containing numpy types) to native Python types."""
    if isinstance(data, np.ndarray):
        return numpy_to_native(data.tolist())  # Converts numpy array to list
    elif isinstance(data, (np.float32, np.float64, np.int32, np.int64)):
        return float(data)  # Converts numpy numbers to Python float
    elif isinstance(data, list):
        return [numpy_to_native(item) for item in data]  # Recursively process each item in the list
    elif isinstance(data, dict):
        return {key: numpy_to_native(value) for key, value in data.items()}  # Process each key-value pair
    else:
        return data  

def main(specific_label, imagenet_path, save_dir, logit_dir, n_layers=24, n_heads=12):

    start_time = time.time()
    print("Starting at ", start_time)
    
    # Get head-by-head ablations
    total_head_level_data = np.empty((n_layers, n_heads), dtype=object)
    for idx, layer_idx in enumerate(range(n_layers)):
        try:
            if idx % 2 == 0: # attn
                for head_idx in range(n_heads):
                    ablation_type = f'layer{layer_idx//2}_head{head_idx}'
                    total_head_level_data[layer_idx, head_idx] = get_logit_diff(ablation_type, specific_label, save_dir, logit_dir)

            else:
                ablation_type = f'layer{layer_idx//2}_mlp'
                total_head_level_data[layer_idx] = get_logit_diff(ablation_type, specific_label, save_dir, logit_dir)
        except Exception as e:
            print(e)
            
    print("Done head-by-head ablations.")
            
    # Get layer-level ablations for attention, instead of head-by-head
    total_layer_level_data = total_head_level_data.copy()
    for idx, layer_idx in enumerate(range(n_layers)):
        try: 
            if idx % 2 == 0: # attn
                ablation_type = f'layer{layer_idx//2}_headNone'
                data = get_logit_diff(ablation_type, specific_label, save_dir, logit_dir)
                total_layer_level_data[layer_idx] = data
            else:
                continue # keep existing mlp data
        except Exception as e:
            print(e)


    data = {
        'head_level': numpy_to_native(total_head_level_data),
        'layer_level': numpy_to_native(total_layer_level_data),
    }

    file_name = f'label_{specific_label}.json'

    with open(os.path.join(save_dir, file_name), 'w') as file:
        json.dump(data, file, indent=4)
        
    print("Saved file to ", os.path.join(save_dir, file_name))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Save logits of all ImageNet datapoints for future analysis.")
    
    parser.add_argument("--specific_label", type=int)
    parser.add_argument("--imagenet_path", type=str, default= '/network/datasets/imagenet.var/imagenet_torchvision/val/', help="Path to the ImageNet dataset.")
    parser.add_argument("--save_dir", type=str, default='/network/scratch/s/sonia.joseph/imagenet_logit_ablation_stats/', help="Directory to save calculated logit differences by class.")
    parser.add_argument("--logit_dir", type=str, default='/network/scratch/s/sonia.joseph/imagenet_logits/', help="Directory to load logits and labels.")

    args = parser.parse_args()
    main(args.specific_label, args.imagenet_path, args.save_dir, args.logit_dir)
    
