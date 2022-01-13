from fastbook import *
from fastai.vision.widgets import *

path = Path('/home/hydra/Downloads/Projekt')

classes = 'ludzie w maskach', 'ludzie bez masek'

data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2,seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)

data = data.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=15.0, min_zoom=1.0, max_zoom=1.3, max_lighting=0.3, max_warp=0.25, p_affine=0.95, p_lighting=0.95, xtra_tfms=None, size=None, mode='bilinear', pad_mode='border', align_corners=True, batch=False, min_scale=1.0))

dls = data.dataloaders(path, bs = 64)

learn = cnn_learner(dls, alexnet, metrics=error_rate)

learn.fit_one_cycle(2)
learn.save('a3.1.1.1')
learn.export(fname="model.pkl")
