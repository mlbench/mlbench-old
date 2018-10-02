def convert_dtype(dtype, obj):
    # The object should be a ``module`` or a ``tensor``
    if dtype == 'fp32':
        return obj.float()
    elif dtype == 'fp64':
        return obj.double()
    else:
        raise NotImplementedError('dtype {} not supported.'.format(dtype))
