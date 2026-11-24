import importlib
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F


class AuthorBrainMRI(nn.Module):
    def __init__(self, source_dir, with_sex=True, heads=8):
        super().__init__()
        source = Path(source_dir).resolve()
        for name in ('MetaBrain.py', 'MetaTransformer.py', 'Module.py'):
            if not (source/name).is_file():
                raise ValueError(f'Missing author source: {name}')
        for name in ('MetaBrain', 'MetaTransformer', 'Module'):
            if name in sys.modules and Path(sys.modules[name].__file__).parent.resolve() != source:
                raise ValueError('Conflicting upstream module already imported')
        sys.path.insert(0, str(source))
        try:
            brain = importlib.import_module('MetaBrain')
            module = importlib.import_module('Module')
        finally:
            sys.path.remove(str(source))
        self.with_sex = with_sex
        self.model = brain.MetaBrainViT(dim_z=256, device=torch.device('cpu'), heads=heads, layer_num=6, dropout=[0., 0.1])
        if with_sex:
            self.model.addInput(brain.MetaBrainInput('sex', module.generateSexInput(dim_z=256), dim_z=256))
        self.model.addOutput(brain.MetaBrainOutput('age', nn.Linear(256, 1)))

    def forward(self, batch):
        image = batch['mri']
        if tuple(image.shape[-3:]) != (192,192,192):
            raise ValueError('Author BrainMRI requires rigid 192-cubed inputs at 1 mm')
        self.model.device = image.device
        inputs = {'mri': image}
        masks = {'mri': torch.zeros(len(image), dtype=torch.bool, device=image.device)}
        if self.with_sex:
            sex = batch['sex']
            if not torch.all((sex == 0) | (sex == 1)):
                raise ValueError('Sex must be explicitly encoded 0/1')
            inputs['sex'] = F.one_hot(sex.long(), 2).float()
            masks['sex'] = torch.zeros_like(masks['mri'])

        return {'age': self.model(inputs, masks)['age'].flatten()*10+65}
