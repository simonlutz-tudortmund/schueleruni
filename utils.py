import os
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from facenet_pytorch import MTCNN, InceptionResnetV1

from google.colab import files

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def upload_file():
  uploaded = files.upload()

  filename = next(iter(uploaded))

  folder = '/content/schueleruni/test_images/Unknown'
  # Check whether the specified path exists or not
  if not os.path.exists(folder):
    # Create a new directory because it does not exist
    os.makedirs(folder)

  shutil.move(os.path.join('/content', filename), os.path.join(folder,filename))


def collate_fn(x):
    return x[0]

mapping={
    'alexandra_popp' : 'Alexandra Popp',
    'angela_merkel' : 'Angela Merkel',
    'barack_obama' : 'Barack Obama',
    'elon_musk' : 'Elon Musk',
    'greta_thunberg' : 'Greta Thunberg',
    'harry_styles' : 'Harry Styles',
    'jennifer_lawrence' : 'Jennifer Lawrence',
    'lebron_james' : 'Lebron James',
    'ryan_goslin' : 'Ryan Goslin',
    'taylor_swift' : 'Taylor Swift',
}

def run_face_recognition():
  # Load face detector
  mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
  )

  # Load facial recognition model
  resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

  # load data
  dataset = datasets.ImageFolder('/content/schueleruni/test_images')
  dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
  loader = DataLoader(dataset, collate_fn=collate_fn)

  # detect faces
  aligned = []
  names = []
  for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

  # compute embedding
  aligned = torch.stack(aligned).to(device)
  embeddings = resnet(aligned).detach().cpu()

  # make predictions
  dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
  data_frame = pd.DataFrame(dists, columns=names, index=names)
  df = data_frame['Unknown']

  predictions = []
  try: 
    df = df.to_frame()
  except:
    pass

  for index, column in df.items():
    prediction = column.drop('Unknown').idxmin()
    predictions.append(prediction)

  # create figure
  fig = plt.figure(figsize=(10, 7))
  plt.subplots_adjust(hspace=0.4)

  # setting values to rows and column variables
  rows = len(predictions)
  columns = 2

  pos = 0
  list = [f for f in os.listdir('/content/schueleruni/test_images/Unknown') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
  pictures = sorted(list)
  for pic in pictures:
    Image1 = mpimg.imread(os.path.join('/content/schueleruni/test_images/Unknown', pic))
    Image2 = mpimg.imread(os.path.join('/content/schueleruni/test_images', predictions[int(pos/2)], '1.jpg'))

    pos += 1
    fig.add_subplot(rows, columns, pos)

    # showing image
    plt.imshow(Image1)
    plt.axis('off')
    plt.title("Original")

    pos += 1
    fig.add_subplot(rows, columns, pos)

    # showing image
    plt.imshow(Image2)
    plt.axis('off')
    plt.title("Vorhersage:\n %s"%(mapping[predictions[int((pos-2)/2)]]))

def clean_files():
  shutil.rmtree('/content/schueleruni/test_images/Unknown')
