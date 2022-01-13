import fastbook
from fastbook import *
from fastai.vision.widgets import *
from IPython.core.pylabtools import figsize

klasy = 'smiling'
path = Path('DaneTreningowe21')
if not path.exists():
  path.mkdir()
  for o in klasy:
    dest = (path/o)
    dest.mkdir(exist_ok=True)
    urls = search_images_ddg(f'people {o}', max_images=50)
    download_images(dest, urls=urls)
