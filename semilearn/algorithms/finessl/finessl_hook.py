
from semilearn.core.hooks import Hook

class FineSSLHook(Hook):
    """
    Pseudo Labeling Hook
    """
    def __init__(self):
        super().__init__()
    
    def before_run(self, algorithm):
        print('FineSSLHook: before_run')

    def after_run(self, algorithm):
        print('FineSSLHook: after_run')

    def after_train_step(self, algorithm):
        print('FineSSLHook: after_train_step')