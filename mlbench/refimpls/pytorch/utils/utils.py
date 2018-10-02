def convert_dtype(options, obj):
    # The object should be a ``module`` or a ``tensor``
    if options.dtype == 'fp32':
        return obj.float()
    elif options.dtype == 'fp64':
        return obj.double()
    else:
        raise NotImplementedError
