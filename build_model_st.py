from PIL import Image
import numpy as np
import os, json
import pandas as pd
import random
import shutil
import streamlit as st
import update_model_st

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

setup_logger()


class main():
    def __init__(self):
        super().__init__()
        st.title('Update model')
        st.write('If it is rlatively simple image, '
                 'total of 1000 - 2000 images are recommended to use to build a new model.')
        form = st.form(key='my_form')
        self.path_to_data = form.text_input(label='Enter the path to the data folder.')  # User input
        self.path_to_model = form.text_input(label='Enter the path where you want to save the model.')  # User input
        self.train_img_ratio = form.text_input(label='Enter the train image ratio to test images. '
                                                     'Initially set to 0.8. (optional)')  # User input
        self.num_workers = form.text_input(label='Enter the number of parallel data loading workers. '
                                                 'Initially set to 2. (optional)')  # User input
        self.img_per_batch = form.text_input(label='Enter the number of images per batch across all machines. '
                                                   'Initially set to 40. (optional)')  # User input
        self.lr = form.text_input(label='Enter the learning rate. Initially set to 0.00025. (optional)')  # User input
        self.max_iter = form.text_input(label='Enter the maximum iteration. Initially set to 4000. (optional)')  # User input
        self.batch_size_per_img = form.text_input(label='Enter the number of proposals to sample for training. '
                                                        'Initially set to 1. (optional)')  # User input
        self.prediction_threshold = form.text_input(label='Enter the threshold of prediction confidence. '
                                                          'Initially set to 0.8. (optional)')  # User input
        submit = form.form_submit_button(label='Submit')
        if submit and self.path_to_data != '' and self.path_to_model != '':
            create_train_test_dataset(self.path_to_data, self.train_img_ratio)
            build_new_model(self.path_to_data, self.path_to_model, self.num_workers, self.img_per_batch, self.lr,
                            self.max_iter, self.batch_size_per_img, self.prediction_threshold)


def length_of_shortest_list(list_of_lists):
    """
    find the length of shortest list in the list.
    :param list_of_lists: list of lists
    :return: (int) the length of shortest list
    """
    lengths = []
    for lst in list_of_lists:
        lengths.append(len(lst))
    return min(lengths)


def create_train_test_dataset(path_to_data, train_img_ratio):
    """
    First, it picks the same number of images from each category.
    Then, it separates those into train and test folders at the specified ratio, train_img_ratio.
    The folders are made in the directory, path_to_data.
    :param path_to_data: path to the folder that contains image and annotation data.
    :param train_img_ratio: ratio of train images to test images. e.g. 0.7: 70% of the total images is train iamges.
    :return: None
    """
    # If there is no user input, use the initial values.
    if train_img_ratio == '':
        train_img_ratio = 0.8
    # Make dictionary to store the image file names for each category.
    filenames_dict = {}
    for file in os.scandir(path_to_data):
        # Only take json files not image files.
        if '.json' in str(file):
            # Open each json file (annotation file).
            with open(file) as jsonfile:
                annot = json.load(jsonfile)
            # Create an item for each category (key: category name, value: list of file names).
            if annot['shapes'][0]['label'] not in filenames_dict.keys():
                filenames_dict[annot['shapes'][0]['label']] = []
            # Append the image file name to the corresponding category list.
            filenames_dict[annot['shapes'][0]['label']].append(annot['imagePath'])

    # The least number of images among all categories.
    shortest_list_length = length_of_shortest_list(list(filenames_dict.values()))
    # The last index of train images in the list.
    train_end_idx = int(round(shortest_list_length * float(train_img_ratio)))
    # Make the directories for test and train data.
    os.mkdir(path_to_data + '\\test')
    os.mkdir(path_to_data + '\\train')
    # Make the test and train dataset.
    for key, filename_list in filenames_dict.items():
        # Shuffle the image names.
        random.shuffle(filename_list)
        # Pick up the image names least numbers of category
        filenames_dict[key] = filename_list[:shortest_list_length]
        # Copy the same numbers of images and json files of each category into train and test folders.
        # Create train dataset.
        for filename in filename_list[:train_end_idx]:
            jsonfilename = filename.split('.')[0] + '.json'
            shutil.copyfile(path_to_data + '\\' + filename, path_to_data + '\\train\\' + filename)
            shutil.copyfile(path_to_data + '\\' + jsonfilename, path_to_data + '\\train\\' + jsonfilename)
        # Create test dataset.
        for filename in filename_list[train_end_idx:shortest_list_length]:
            jsonfilename = filename.split('.')[0] + '.json'
            shutil.copyfile(path_to_data + '\\' + filename, path_to_data + '\\test\\' + filename)
            shutil.copyfile(path_to_data + '\\' + jsonfilename, path_to_data + '\\test\\' + jsonfilename)
    # Create the coco format annotation file from the labelme files.
    update_model_st.convert_labelme_to_coco(path_to_data + '\\train\\')
    update_model_st.convert_labelme_to_coco(path_to_data + '\\test\\')


def build_new_model(path_to_data, path_to_model, num_workers, img_per_batch, lr, max_iter, batch_size_per_img,
                    prediction_threshold):
    """
    Build a new model and create the validation report using the test data.
    :param path_to_data: (string) Path to the folders that contains the train and test folders.
    :param path_to_model: (string)Path to the folder whree the model will be saved.
    :param num_workers: (int, optional) Number of parallel data loading workers. If none, 2.
    :param img_per_batch: (int, optional) Number of images per batch across all machines. This is also the number
    of training images per step (i.e. per iteration). If we use 16 GPUs and IMS_PER_BATCH = 32,
    each GPU will see 2 images per batch. If none, 1.
    :param lr: (float, optional) Learning rate. If none, 0.00025.
    :param max_iter: (int, optional) Maximum iteration. If none, 4000.
    :param batch_size_per_img: (int, optional) Number of proposals to sample for training. If none, 40.
    :param prediction_threshold: (float, optional)Threshold of prediction confidence. If none, 0.8.
    :return: output folder that contains the new model with the model information will be saved in the path_to_model.
    """
    # If there is no user input, use the initial values.
    if num_workers == '':
        num_workers = 2
    if img_per_batch == '':
        img_per_batch = 1
    if lr == '':
        lr = 0.00025
    if max_iter == '':
        max_iter = 4000
    if batch_size_per_img == '':
        batch_size_per_img = 40
    if prediction_threshold == '':
        prediction_threshold = 0.8
    st.write('Number of workers: ' + num_workers)
    st.write('Image per batch: ' + img_per_batch)
    st.write('Learning rate: ' + lr)
    st.write('Maximum iteration: ' + max_iter)
    st.write('Batch size per image: ' + batch_size_per_img)
    st.write('Threshold for the prediction confidence: ' + prediction_threshold)

    # Open the coco format annotations.
    with open(path_to_data + r'\\test\coco_annotation.json') as jsonfile:
        coco_d = json.load(jsonfile)
    # Create the category information. The keys are category names and the values are their IDs.
    categories = {}
    for cat in coco_d['categories']:
        categories[cat['name']] = cat['id']

    # Create the dictionary that contains the file names and category IDs.
    correct_classification = {}
    for annot in coco_d['annotations']:
        for img in coco_d['images']:
            if img['id'] == annot['image_id']:
                correct_classification[os.path.basename(img['file_name'])] = annot['category_id']
                break

    # Register the training and testing data.
    register_coco_instances("coco_train", {}, path_to_data + r'\\train\coco_annotation.json', path_to_data + r'\\train')
    register_coco_instances("coco_test", {}, path_to_data + r'\\test\coco_annotation.json', path_to_data + r'\\test')
    for d in ["train", "test"]:
        MetadataCatalog.get("meta_" + d).set(thing_classes=categories)
    coco_test_dataset = DatasetCatalog.get("coco_test")

    # Build a new model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = path_to_model
    cfg.DATALOADER.NUM_WORKERS = int(num_workers)
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = int(img_per_batch)
    cfg.SOLVER.BASE_LR = float(lr)
    cfg.SOLVER.MAX_ITER = int(max_iter)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(batch_size_per_img)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(str(prediction_threshold))   # set a custom testing threshold
    cfg.DATASETS.TEST = ('coco_test',)
    cfg.TEST.EVAL_PERIOD = 100
    predictor = DefaultPredictor(cfg)

    # Test the model.
    result_dict = {}
    for d in coco_test_dataset:  # random.sample(coco_k_5['images'], 5):
        path = d['file_name']
        im = Image.open(path)
        im_array = np.asarray(im)
        outputs = predictor(im_array)
        instances = outputs["instances"]
        if len(instances) > 0:
            highest_score_instances = instances[instances.scores == instances.scores.cpu().data.numpy().max()]
            result_dict[os.path.basename(path)] = int(highest_score_instances.pred_classes[0].to("cpu").numpy())
        else:
            result_dict[os.path.basename(path)] = 'not annotated'
    # Make the result DataFrame.
    result_df = pd.DataFrame({'Prediction': result_dict.values()}, index=result_dict.keys())
    result_df.index.name = 'File Name'
    correct_classification_df = pd.DataFrame({'Correct Classification': correct_classification.values()},
                                             index=correct_classification.keys())
    correct_classification_df.index.name = 'File Name'
    validation_report = pd.merge(correct_classification_df, result_df, on='File Name')
    # Validate the accuracy of predictions.
    correct_pred_count = 0
    validation_report['Validation'] = ''
    for i in validation_report.index:
        if validation_report.loc[i, 'Correct Classification'] == validation_report.loc[i, 'Prediction']:
            validation_report.loc[i, 'Validation'] = ''
        elif validation_report.loc[i, 'Correct Classification'] != validation_report.loc[i, 'Prediction']:
            validation_report.loc[i, 'Validation'] = 'Incorrect prediction'
            correct_pred_count += 1
    # Calculate the accuracy of the model.
    accuracy = correct_pred_count/len(validation_report.index)*100
    accuracy_df = pd.DataFrame({'Accuracy (%)': [accuracy]}, index = ['Accuracy (%)'])
    # validation_report['Accuracy (%)'] = accuracy
    # validation_report.to_excel(path_to_model + r'\validation_report.xlsx')
    st.write('Accuracy: ' + str(accuracy) + ' %')

    excel_path = path_to_model + r'\validation_report.xlsx'
    excelwriter = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    excelsheet_names = ['Accuracy', 'Validation']
    # Make all the excel sheets.
    for i, df in enumerate([accuracy_df, validation_report]):
        df.to_excel(excelwriter, sheet_name=excelsheet_names[i], index=True)
    # Save the file
    excelwriter.save()
