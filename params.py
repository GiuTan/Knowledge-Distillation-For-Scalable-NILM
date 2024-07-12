params = {
          'strong_weakUK': { 'drop': 0.1,
                         'kernel': 5,
                         'layers': 3,
                          'GRU': 64,
                          'cs': False,
                            'no_weak' : False,
                          'pre_trained': 'pretrained_UKDALE.h5'}, #
          'strong_weakREFIT': { 'drop': 0.1,
                             'kernel': 5,
                             'layers': 3,
                              'GRU': 64,
                              'cs': False,
                              'no_weak' : False,
                              'pre_trained': 'pretrained_REFIT.h5'
              }}

uk_params = {'mean': 453.57,
            'std': 729.049}
refit_params = {'mean': 537.051,
                'std': 746.905}

epochs = 1000
classes = 6
batch_size = 64
window_size = 2550