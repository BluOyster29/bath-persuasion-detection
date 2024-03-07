from utils import getargs, load_config
from preprocess import preprocess_data






def main():
    args = getargs()
    config = load_config(args.config)
    training_dataloader, testing_dataloader = preprocess_data(
        config)

    output(
        training_dataloader,
        f'{config.get("dataloader_path")}training_dataloader.pkl'
    )

    output(
        testing_dataloader,
        f'{config.get("data_loader_path")}test_dataloader.pkl'
    )

    return


if __name__ == '__main__':
    main()
