
# In order to fix an error with some images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def learn_from_dir(method_classifier,
                   dir_dataset,
                   root_checkpoint,
                   iteration_resume,
                   number_epochs,
                   learning_rate,
                   batch_size,
                   size_image,
                   is_random,
                   perc_dropout,
                   betas,
                   gpu_id,
                   prop_training,
                   number_workers
                   ):
    pass

def test_from_dir(name_classifier,
                  dir_dataset_test,
                  batch_size,
                  target_size,
                  rescale
                  ):
    pass