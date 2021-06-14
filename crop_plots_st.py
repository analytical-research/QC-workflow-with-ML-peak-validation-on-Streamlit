import cv2
import numpy as np
import os
import pandas as pd
import qc_data_process_conc_calc
import streamlit as st

from pdf2image import convert_from_path


# This code crops the pdf result and make UV and CAD plot images. Save the pdfs as the following:
# (Folder -> WK## -> Mnova result folder as named by Mnova -> pdf -> pdf files)
# The images will be named as CAD-WK##-N-Well.jpg and saved in cropped_img folder that is created in Folder.
# N is the number starting from 0 added in the chronological order as it process through the sample ID folder.
# Both CAD and UV plots are mixed.

# Crop plots
class main():
    def __init__(self):
        super().__init__()
        st.title('Crop plots')
        st.write('The PDF files have to be saved as the following:')
        st.write('Folder -> WK## -> Mnova result folders as named by Mnova -> pdf -> pdf files')
        form = st.form(key='my_form')
        self.parent_dir = form.text_input(label='Enter the path to the data folder.')
        submit = form.form_submit_button(label='Submit')
        if submit:
            # st.write("mnovafolder_path")
            # st.write(type(self.mnovafolder_path))
            # st.write(self.mnovafolder_path)
            create_img_dataset(self.parent_dir)


def create_img_dataset(path):
    """
    Create and save the cropped image dataset from the Mnova pdf.
    :return:
    """
    # Create the directory, cropped_img, where the images will be saved.
    dst = os.path.join(path, 'cropped_img')
    os.makedirs(dst)

    # Loop through the folder, WK##, CD##, and etc.
    for sample_folder in os.scandir(path):
        if os.path.basename(sample_folder) != 'cropped_img':
            # Get the sample ID from the folder name.
            sampleid = os.path.basename(sample_folder)
            # Loop through the sample ID folder that contains the Mnova result folders.
            for i, Mnova_result_folder in enumerate(os.scandir(sample_folder)):
                mnova_folder_num = str(i)
                file_name_list = []  # List for pdf file names.
                pdf_path = os.path.join(Mnova_result_folder, 'pdf')
                # Loop through the pdf files in the folder named as pdf.
                for file in os.listdir(pdf_path):
                    if 'pdf' in file:
                        # Get the file name without extension is appended to the list.
                        file_name = os.path.basename(file)
                        file_name = file_name.split('.')[0]
                        file_name = mnova_folder_num + '-' + file_name[-3:]
                        file_name_list.append(file_name)

                # Create the pd.DataFrame of pdf file names.
                file_name_df = pd.DataFrame({'Well': file_name_list, 'G#': file_name_list})
                file_name_df.set_index('G#')

                # Crop the CAD and UV plot images.
                cropped_img_df = crop_uvcad_plots(pdf_path, file_name_df)

                # Name and save the UV images in the folder named cropped_img.
                for j, img in enumerate(cropped_img_df['UVpeaks']):
                    # Save the image into the cropped_img folder
                    cv2.imwrite(dst + '\\' + 'UV-' + sampleid + '-' + cropped_img_df.loc[:, 'Well'][j] + '.jpg', img)

                # Name and save the CAD images in the folder named cropped_img.
                for j, img in enumerate(cropped_img_df['CADpeaks']):
                    # Save the image into the cropped_img folder
                    cv2.imwrite(dst + '\\' + 'CAD-' + sampleid + '-' + cropped_img_df.loc[:, 'Well'][j] + '.jpg', img)


def crop_uvcad_plots(path_to_pdfs, filename_df):
    """
    This crops the UV and CAD plots on the result pdf removing the shaded parts.
    The thresholds (shaded parts) can be set on Mnova.
    :param path_to_pdfs: Path to the folder that contains the pdf files.
    :param filename_df: pd.DataFrame contains Well. The index is G#.
    :return: pd.DataFrame contains Well, G#, UVpeaks (np.array), and CADpeaks (np.array). Well is the index.
    """
    # Cropped images will be saved in the dictionaries.
    # The first dict is for UV images and the second dict is for CAD images.
    # The keys are N - file name, 0-A01.
    # N is the number starting from 0 added in chronological order as it processes through the sample ID folder.

    # Peak images are saved in the following format.
    # The first dictionary is for the UV peaks and the second is for the CAD peaks.
    peaks_lists = [{}, {}]
    # X = []
    # count = 0
    # Loop through the pdf folder.
    for file in os.scandir(path_to_pdfs):
        filename = os.path.basename(file)
        # Ensure to pick up only the pdf files with the usual name, 1-A01.pdf or A01.pdf...
        if '.pdf' in filename and '(' not in filename:
            # Get file name without extension.
            filename_wo_ext = os.path.splitext(filename)[0]
            # Convert from pdf to image file.
            pdfs = convert_from_path(file)
            # Get only the image
            pdf = pdfs[0]

            # Two cropping locations. From the top left corner,
            # (upper left corner x, upper left corner y, bottom right corner x, bottom right corner y)
            uv_im_crop_loc = (197, pdf.size[1] / 2 + 190, pdf.size[0] - 45, pdf.size[1] / 2 + 365)
            cad_im_crop_loc = (212, pdf.size[1] - 332, pdf.size[0] - 45, pdf.size[1] - 138)

            # Crop the image at the locations specified above.
            uv_im = pdf.crop(uv_im_crop_loc)
            cad_im = pdf.crop(cad_im_crop_loc)

            # convert image into numpy array.
            uv_na = np.array(uv_im)
            cad_na = np.array(cad_im)

            # convert image into gray scale and into np.array.
            uv_na_gray = np.array(uv_im.convert("L"))
            cad_na_gray = np.array(cad_im.convert("L"))

            # Get the X and Y positions where the color is gray (the discarding area).
            uv_y, uv_x = np.where(uv_na_gray == 212)
            cad_y, cad_x = np.where(cad_na_gray == 212)

            # get the x positions array of gray area at Y = 0 using the gray image. This is same for all Ys.
            uv_xpos_at_y0 = uv_x[0:int(len(uv_x) / uv_na_gray.shape[0])]
            cad_xpos_at_y0 = cad_x[0:int(len(cad_x) / cad_na_gray.shape[0])]
            list_xpos_at_y0 = [uv_xpos_at_y0, cad_xpos_at_y0]
            na_images = [uv_na, cad_na]

            # Loop through the x positions at Y = 0.
            for k, xpos_at_y0 in enumerate(list_xpos_at_y0):
                cropped_na = na_images[k]
                # Crop only the image within the thresholds (white area).
                # Check if the gray area is continuous throughout.
                if qc_data_process_conc_calc.check_consecutive(xpos_at_y0):
                    # Gray area is only on one side.
                    if xpos_at_y0[0] == 0:
                        # Gray area is on the left side.
                        xpos = xpos_at_y0[-1]
                        cropped_na = na_images[k][:, xpos + 1:, :]
                    elif xpos_at_y0[0] != 0:
                        # Gray area is on the right side.
                        xpos = xpos_at_y0[0]
                        cropped_na = na_images[k][:, 0: xpos, :]
                # Check if the gray area is not continuous.
                elif not qc_data_process_conc_calc.check_consecutive(xpos_at_y0):
                    # Gray areas are on both sides.
                    count = xpos_at_y0[0]

                    # Loop through the x positions at Y = 0.
                    # The end of the left gray area and the start position of the right gray area are
                    # when the x positions are not continuous.
                    for i, pos in enumerate(xpos_at_y0):
                        if count == pos:
                            count += 1
                            continue
                        else:
                            first_xpos = xpos_at_y0[i - 1]
                            second_xpos = xpos_at_y0[i]
                            # Crop the white area between the gray areas.
                            cropped_na = na_images[k][:, first_xpos:second_xpos, :]
                            break

                # Dimension of the image size. The image size has to be same for all.
                dim = (700, 200)
                # Resize and save the image in a dictionary: key = file name, value = image.
                for well in filename_df['Well']:
                    if filename_wo_ext[-3:] in well[-3:]:
                        peaks_lists[k][well] = cv2.resize(cropped_na, dim, interpolation=cv2.INTER_AREA)
                        #
                        # X.append([cropped_na])
                        # count = count + 1
                    else:
                        pass

    # Create the pd.DataFrame with UV and CAD images. The index is the well numbers.
    cropped_im_df = pd.DataFrame({'UVpeaks': list(peaks_lists[0].values()), 'CADpeaks': list(peaks_lists[1].values()),
                                 'Well': list(peaks_lists[0].keys())})
    cropped_im_df.set_index('Well')

    # Merge the file name DataFrame with the image DataFrame horizontally using the Well numbers.
    cropped_im_df = filename_df.merge(cropped_im_df, how='left', on='Well')

    return cropped_im_df

