#!/usr/bin/env python3
def initialize_attributes(cls):
    """
    class decorator
    function:
        --initialize some class attributes
    """
    def wrapper(*args, **kwargs):
        instance = cls(*args, **kwargs)
        instance.run()

        return instance

    return wrapper
