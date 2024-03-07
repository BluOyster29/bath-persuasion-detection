import logging 
from preprocess import import_data
from build_dataloaders import build_dataloaders
from train import train
from utils import get_args, load_config
import warnings 

warnings.filterwarnings("ignore", category=FutureWarning) 
# until i work out pandas bug 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(parsed_args):
    
    logger.info('Loading config from %s', parsed_args.config_path)
    config = load_config(parsed_args.config_path)
    logger.info('Config Loaded')
    logger.info('Importing Data')
    train_df, test_df, label_columns = import_data(config)
    training_dataloader, testing_dataloader = build_dataloaders(train_df[:1], test_df[:1], config)
    trained_model = train(config, label_columns, training_dataloader)

if __name__ == '__main__':
    logger.info('Starting Main')
    args = get_args()
    main(args)