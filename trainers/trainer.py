import pytorch_lightning as pl

class Trainer(pl.Trainer):
    def __init__(self, cfg, model):
        super(Trainer, self).__init__(cfg)
        self.cfg = cfg
        self.model = model
    
    