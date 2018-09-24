def create_dataset(options, train=True):
    if options.libsvm_dataset:
        from .load_libsvm_dataset import create_dataset
    else:
        from .load_dataset import create_dataset

    return create_dataset(options, train)
