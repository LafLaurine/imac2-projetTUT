import torchvision.datasets as dset
import torchvision.transforms as transforms


class ImageFolderCapsule(dset.ImageFolder):
    """
    A custom data loader where classes labels
    are set according to config.
    Basically copies DatasetFolder but changes 'samples' field
    """
    # __dict_labels
    def __init__(self, classifier, *args, **kwargs):
        self.__dict_labels = classifier.get_dict_classes()
        super(ImageFolderCapsule, self).__init__(*args, **kwargs)
        print(self.class_to_idx)

    """
    wild overload for custom classes
    """
    # @overload
    def _find_classes(self, dir):
        classes = self.__dict_labels.keys()
        class_to_idx = self.__dict_labels
        return classes, class_to_idx


def get_transform(size_image):
    transform_fwd = transforms.Compose([
        transforms.Resize((size_image, size_image)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform_fwd
