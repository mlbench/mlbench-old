from utils.parser import MainParser
from config import initialize


def main():
    parser = MainParser()
    options = parser.parse_args()
    options = initialize(options)

    print(options)


if __name__ == '__main__':
    main()
