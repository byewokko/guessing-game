import typing


def dict_default(d: dict,
                 key: typing.Hashable,
                 default: typing.Any = None,
                 function: typing.Callable = None,
                 warning: bool = False):
    if key in d.keys():
        if function:
            return function(d[key])
        return d[key]
    else:
        if warning:
            print(f"'{key}' not found in {d.__name__}. Defaulting to {default}.")
        return default
