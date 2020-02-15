import torchvision.datasets as dset


from ...common_config import match_labels_dict

class ImageFolderCapsule(dset.ImageFolder):
    def __init__(self, classifier, *args, **kwargs):
        super(ImageFolderCapsule, self).__init__(*args, **kwargs)
        self.set_labels_idx(classifier)

    def set_labels_idx(self, classifier):
        match_labels_dict(self.class_to_idx, classifier.get_classes())
