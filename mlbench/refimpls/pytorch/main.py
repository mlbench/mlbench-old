from utils.parser import MainParser


def main():
    parser = MainParser()
    options = parser.parse_args()
    print(options)


if __name__ == '__main__':
    main()
