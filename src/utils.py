import argparse 
import yaml 
import pickle

def get_args():
    # Generate Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config.yaml")
    return parser.parse_args()


def load_config(config_path):
    # Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def output_pickle(object, config, item_name):
    # Output Pickle
    with open(f"{config.get('output_dataloader')}_{item_name}.pkl", 'wb') as f:
        pickle.dump(object, f)
        
def gen_metadata(config, metatype, **kwargs):
    """
    Generate metadata based on the given configuration and metatype.

    Args:
        config (dict): A dictionary containing configuration parameters.
        metatype (str): The type of metadata to generate.

    Returns:
        str: The generated metadata.

    """
    if metatype == 'datalodaer':
        meta = f"{config.get('batch_size')}_{config.get('max_token_len')}_{config.get('sampler')}"
    
    if metatype == 'model':
        meta = f"{config.get('pretrained_model')}_{config.get('num_epochs')}_{config.get('learning_rate')}_{config.get('batch_size')}"
    return meta
