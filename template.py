"""
Network configuration similar to the original paper
Here, same number of features used for all parts
In the paper, half number of features are used for the base block of the model
"""

def get_nas_config(quality):
    if quality == 'low':
        return {4: {'block' : 8, 'feature' : 9, 'output_filter': 2},
                3: {'block' : 8, 'feature' : 8, 'output_filter': 2},
                2: {'block' : 8, 'feature' : 4, 'output_filter': 2},
                1: {'block' : 1, 'feature' : 2, 'output_filter': 1}}
    elif quality == 'medium':
        return {4: {'block' : 8, 'feature' : 21, 'output_filter': 2},
                3: {'block' : 8, 'feature' : 18, 'output_filter': 2},
                2: {'block' : 8, 'feature' : 9, 'output_filter': 2},
                1: {'block' : 1, 'feature' : 7, 'output_filter': 1}}
    elif quality == 'high':
        return {4: {'block' : 8, 'feature' : 32, 'output_filter': 2},
                3: {'block' : 8, 'feature' : 29, 'output_filter': 2},
                2: {'block' : 8, 'feature' : 18, 'output_filter': 2},
                1: {'block' : 1, 'feature' : 16, 'output_filter': 1}}
    elif quality == 'ultra':
        return {4: {'block' : 8, 'feature' : 48, 'output_filter': 2},
                3: {'block' : 8, 'feature' : 42, 'output_filter': 2},
                2: {'block' : 8, 'feature' : 26, 'output_filter': 2},
                1: {'block' : 1, 'feature' : 26, 'output_filter': 1}}
    else:
        raise NotImplementedError
