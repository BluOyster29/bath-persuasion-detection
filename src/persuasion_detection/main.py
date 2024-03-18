import logging
import sys
from preprocess import import_data
from build_dataloaders import build_dataloaders
from train import train
from evaluate import evaluate
from utils import get_args, load_config
import warnings

sys.path.append('../models')
warnings.filterwarnings("ignore", category=FutureWarning)
# until i work out pandas bug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_data(config):
    """
    Fetch the data from the specified data source.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        pd.DataFrame: The training data.
        pd.DataFrame: The testing data.
    """

    logger.info('Fetching Data')
    train_df, test_df, vocab = import_data(config)
    logger.info('Data Fetched')
    return train_df, test_df, vocab


def main(parsed_args):

    config = load_config(parsed_args.config_path)
    train_df, test_df, vocab = fetch_data(config)

    training_dataloader, testing_dataloader = build_dataloaders(
        train_df, test_df, config, vocab)

    trained_model = train(config, training_dataloader)

    evaluate(config, testing_dataloader, trained_model)


if __name__ == '__main__':
    logger.info('Starting Main')
    args = get_args()
    main(args)
