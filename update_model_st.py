import json
import labelme2coco
import numpy as np
import os
import random
import streamlit as st

from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

setup_logger()


# UpdateModel
class main():
    def __init__(self):
        super().__init__()
        st.title('Update model')

        form = st.form(key='my_form')
        self.data_folder_path = form.text_input(label='Enter the path to the data folder.')
        self.model_path = form.text_input(label='Enter the path to the model.')
        submit = form.form_submit_button(label='Submit')
        if submit:
            # st.write("mnovafolder_path")
            # st.write(type(self.mnovafolder_path))
            # st.write(self.mnovafolder_path)
            # Convert labelme annotations to coco format and validate the annotations.
            convert_labelme_to_coco(path_to_data=self.data_folder_path)
            update_model(path_to_data=self.data_folder_path, path_to_model=self.model_path)


def convert_labelme_to_coco(path_to_data):
    """
    Convert the labelme annotations to coco format and validate the annotations with randomly picked 5 images.
    If the annotation is incorrect, the program terminates.
    :param path_to_data: Path to the folder that contains the image and annotation data.
    :return: None
    """
    # convert labelme annotations to coco
    labelme2coco.convert(path_to_data, path_to_data + r'\coco_annotation.json')

    # Open the coco format data
    with open(path_to_data + r'\coco_annotation.json') as f:
        coco_d = json.load(f)

    # Get the category IDs for each category and create a new "categories" section.
    categories = []
    # for category in coco_d['categories']:
    #     if category['name'] == 'Bad':
    #         categories.append({"id": category['id'],
    #                            "name": category['id'],
    #                            "supercategory": category['id'],
    #                            "isthing": 1,
    #                            "color": [222, 23, 1]
    #                            })
    #     elif category['name'] == 'Good':
    #         categories.append({"id": category['id'],
    #                            "name": "Good",
    #                            "supercategory": "Good",
    #                            "isthing": 1,
    #                            "color": [133, 23, 1]
    #                            })

    # Update the "catogories" section of the coco format data with the correct category IDs.
    # coco_d['categories'] = categories

    categories = []
    for cat in coco_d['categories']:
        cat['isthing'] = 1
        categories.append(cat['name'])

    # Fix the segmentation and bbox.
    for annot in coco_d['annotations']:
        annot['bbox_mode'] = 0
        seg = annot['segmentation'][0]
        annot['bbox'] = seg
        annot['segmentation'] = [[seg[0], seg[1], seg[0], seg[3], seg[2], seg[3], seg[2], seg[1]]]

    # Save the modified coco format data.
    with open(path_to_data + r'\coco_annotation.json', 'w') as j:
        json.dump(coco_d, j, sort_keys=True, indent=4)

    # Show the images to the user to validate the annotations.
    # Register the image information.
    register_coco_instances("coco_visualise", {}, path_to_data + r"/coco_annotation.json",
                            path_to_data)
    MetadataCatalog.get("meta_visualise").set(thing_classes=categories)
    # MetadataCatalog.get("meta_train").set(thing_classes=["Bad", "Good"], thing_colors=[(172, 0, 0), (229, 0, 0)])
    train_metadata = MetadataCatalog.get("meta_visualise")
    coco_train_dataset = DatasetCatalog.get("coco_visualise")

    st.write('Showing the randomly picked 5 images. Check if the annotation is correctly embedded.')
    # Randomly pick 5 images to show to the user to validate the annotations.
    for d in random.sample(coco_train_dataset, 5):
        im = Image.open(d['file_name'])
        im_array = np.asarray(im)
        v = Visualizer(im_array, metadata=train_metadata, instance_mode=ColorMode.SEGMENTATION, scale=0.5)
        v = v.draw_dataset_dict(d)
        pil_image = Image.fromarray(v.get_image())
        st.image(pil_image)
        # window = tk.Toplevel()
        # window.tkimage = ImageTk.PhotoImage(pil_image)
        # window.attributes('-topmost', True)
        # label = tk.Label(window, image=window.tkimage)
        # label.pack()
        # button_close = tk.Button(window, text="Close", command=window.destroy)
        # button_close.pack(fill='x')

    # Confirm the annotations with user. If the annotations are correct, it will proceed further.
    # If not, it terminates the program.
    # if messagebox.askyesno(title="Validate Annotations", message="Were all annotations correct?"):
    #     pass
    DatasetCatalog.clear()
    MetadataCatalog.clear()

def update_model(path_to_data, path_to_model):
    """
    Update the prediction model with the new data.
    :param path_to_data: Path to the folder that contains the image and coco format annotation data.
    :param path_to_model: Path to the folder that contains the model.
    :return: None
    """
    # Open the annotation files.
    with open(path_to_data + r'\coco_annotation.json') as f:
        coco_d = json.load(f)
    # Get the categories.
    categories = []
    for cat in coco_d['categories']:
        categories.append(cat['name'])

    # Register the new data.
    register_coco_instances("coco_update", {}, path_to_data + r"\coco_annotation.json", path_to_data)
    MetadataCatalog.get("meta_update").set(thing_classes=categories)
    # MetadataCatalog.get("meta_update").set(thing_classes=["Bad", "Good"], thing_colors=[(172, 0, 0), (229, 0, 0)])

    # Set the parameters.
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("coco_update",)
    cfg.OUTPUT_DIR = path_to_model
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 400
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 10
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)

    # Update the model.
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

