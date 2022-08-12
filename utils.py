import numpy as np
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from object_detection.utils import visualization_utils as viz_utils

def preprocess_bboxes(bboxes,image):
  
  '''
  Preprocess bounding boxes 
  
  Args:
    bboxes: Array of [N_i,4]
    image: Array of (H,W,3)
  
  Return:
    Standarized bounding boxes
  '''
  
  shape=image.shape
  y_factor=shape[0]
  x_factor=shape[1]
  shape_img=np.array([y_factor,x_factor,y_factor,x_factor])
  bboxes/=shape_img
  return bboxes

def reshape_img_and_bboxes(img,bboxes,target_size):
  '''
  Reshape an image and its bboxes
  
  Args:
    img: np.array of (H,W,3)
    bboxes: np.array of (N_i,4)
    target_size: tuple of two elements.
  
  '''
  
  y_,x_=img.shape[0],img.shape[1]
  y_scale=target_size[0]/y_
  x_scale=target_size[1]/x_
  bboxes=bboxes*np.array([y_,x_,y_,x_])
  bboxes_=bboxes*np.array([y_scale,x_scale,y_scale,x_scale])
  bboxes_/=np.array([target_size[0],target_size[1],target_size[0],target_size[1]])
  img_=tf.image.resize(img,target_size).numpy()
  return img_,bboxes_

def get_train_step_func(model,vars_to_tune,optimizer):
  '''
  Get the train_step_func.
  
  Args:
  
    model: Instance of model from tensorflow_garden
    vars_to_tune: Vars of the model wich will be trained
    optimizer: Instance of tf.keras.optimizers.Optimizer
  
  Return:
    return a train_ste_func
  '''
  
  @tf.function
  def train_step_func(images,true_bboxes,true_labels):
    '''
    Apply the gradients and get the loss.
    
    Args:
    
      true_boxes: list of arrays
      true_labels: list of labels in one_hot
    
    Returns:
      return total loss in a batch
    '''
    model.provide_groundtruth(
        groundtruth_boxes_list=true_bboxes,
        groundtruth_classes_list=true_labels
    )
    with tf.GradientTape() as tape:
      images,shapes=model.preprocess(images)
      prediction_dict = model.predict(images, shapes)
      losses_dict=model.loss(prediction_dict,shapes)
      total_loss=losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
      
      grads=tape.gradient(total_loss,vars_to_tune)
      optimizer.apply_gradients(zip(grads,vars_to_tune))

    return total_loss  
  return train_step_func  

def get_val_func(model):
  
  '''
  Get the validation_func
  
  Args:
    model: Instance of model from tensorflow_garden

  '''
  
  @tf.function
  def validation_func(images,true_bboxes,true_labels):
    
    '''
    Get the loss on a batch.
    
    Args:
      true_boxes: list of arrays [1,4]
      true_labels: list of labels in one_hot [1, num_classes]
    
    Return:
      return total loss in a batch
    
    '''

    model.provide_groundtruth(
        groundtruth_boxes_list=true_bboxes,
        groundtruth_classes_list=true_labels
    )
    images,shapes=model.preprocess(images)
    prediction_dict = model.predict(images,shapes)
    losses_dict=model.loss(prediction_dict,shapes)
    total_loss=losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

    return total_loss
  return validation_func  

def train_on_ds(train_func,train_ds):
  '''
  Apply the train_func in all the batches
  
  Args:
    train_func: function wich returns the loss and apply the gradients in some data.
    train_ds: list of tensors (img,bboxes,labels)
  
  Return:
    the average loss
  '''
  
  
  size_ds=len(train_ds)
  total_loss=0
  for k in range(size_ds):
    img,bboxes,labels=train_ds[k]
    labels=labels[:,np.newaxis]
    bboxes=[bboxes[i][np.newaxis,:] for i in range(bboxes.shape[0])]
    labels=[labels[i] for i in range(labels.shape[0])]
    loss=train_func(img,bboxes,labels)
    total_loss+=loss.numpy()

  return total_loss/size_ds

def val_on_ds(val_func,val_ds):

  '''
    Apply the val_func in all the batches
  
    Args:
      val_func: function wich returns the loss and apply the gradients in some data.
      train_ds: list of tensors (img,bboxes,labels)
  
    Return:
      the average loss
  
  '''
  size_ds=len(val_ds)
  total_loss=0
  for k in range(size_ds):
    img,bboxes,labels=val_ds[k]
    labels=labels[:,np.newaxis]
    bboxes=[bboxes[i][np.newaxis,:] for i in range(bboxes.shape[0])]
    labels=[labels[i] for i in range(labels.shape[0])]
    loss=val_func(img,bboxes,labels)
    total_loss+=loss.numpy()
  return total_loss/size_ds  


def create_frame_video(video_path,name_frame,frames_path):
  '''
  Creates a frames videos from a video
  
  Args:
    video_path: Path of the video
    frames_path: Path where the frames will be
  '''
  
  vidcap = cv2.VideoCapture(video_path)
  success,image = vidcap.read()
  count = 0
  while success:
    name=name_frame+"_"+('%05d' % count)+".jpg"
    cv2.imwrite(join(frames_path,name),image)     
    success,image = vidcap.read()
    print(f'Read a new frame, name: {name} state:{success}')
    count += 1


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    min_score=0.3,
                    image_name=None):
  """
  Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=min_score)
  image_np_with_annotations/=255.0 
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)
 

def create_images_postprocess(model,validation_images,images_path,fig_size,min_score,category_index):
  '''
  Create predictions from a object detection model
  
  Args:
    model: Instance of model from tensorflow_garden
    validation_images: List of np.array (H,W,3)
    images_path: Path where the predictions images will be saved.
    fig_size: Size of the images.
    category_index: Dict of categories.

  '''
  
  for i,img in enumerate(validation_images):
    img_=tf.expand_dims(img,axis=0)
    img_,shape=model.preprocess(img_)
    detections=model.predict(img_,shape)
    detections=model.postprocess(detections,shape)
    plot_detections(
        img,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.uint32),
        detections['detection_scores'][0].numpy(),
        category_index,figsize=fig_size,min_score=min_score,image_name=images_path+"/POST_FRAME_"+('%05d' % i)+".jpg"
    )

def create_array_from_images_path(images_path,name_frame,target_size,size_min=1):
  '''
  Creates a list of images(array of [H,W,3]) 
  
  Args:
    images_path: Path where the images are saved from video
    name_frame: Name of the frame before '_'
    size_min: The min size of the images that will be used (1 is the max)
  
  Returns:
    List of np.array
  '''
  assert size_min<=1.0
  images=[]
  glob_images=sorted(glob(images_path+'/'+name_frame+'_'+'*.jpg'))
  len_size=int(size_min*len(glob_images))
  glob_images=glob_images[:len_size]
  for name_img in glob_images:
    img_PIL=load_img(join(images_path,name_img))
    img_np=img_to_array(img_PIL)
    img_np=tf.image.resize(img_np,target_size).numpy()
    images.append(img_np)
  return images
  
def create_video(images,name_video):
  
  '''
  Creates a video from frames
  
  images: list of arrays [H,W,3]
  name_video: name of the video that will be .mp4
  
  '''
  
  height,width,depth=images[0].shape
  size=(width,height)

  out = cv2.VideoWriter(name_video+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 50, size)
 
  for i in range(len(images)):
    out.write(images[i].astype('uint8'))
  out.release()

def make_category_index(list_classes):
  '''
  Creates a category index for using viz_utils
  
  Args:
    list_classes: List of classes in order
  
  Return:
    Dictionary with ids and names
  
  '''
  
  num_classes=len(list_classes)
  dict_={}
  for i in range(num_classes):
    dict_[i]={'id':i,'name':list_classes[i]}
  return dict_

classes=['N', 'C', 'D', 'T', 'W']
category_index=make_category_index(classes)

