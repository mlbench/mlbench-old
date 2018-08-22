from config import initialize

from utils.parser import MainParser
from datasets.load_dataset import create_dataset


def main():
    parser = MainParser()
    options = parser.parse_args()
    options = initialize(options)

    options = create_dataset(options, train=True)
    options = create_dataset(options, train=False)


if __name__ == '__main__':
    main()
