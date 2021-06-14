import numpy as np
import os
import pandas as pd
import streamlit as st

from sklearn import linear_model


std_info_path = r"Y:\Data\2021\Data Process Templates\STD_template\std_masses.xlsx"
std_path = r"Y:\Data"
threshold_percent_diffs = 20
num_val_rep = 3


# STDValidation
class main():
    def __init__(self):
        super().__init__()
        st.title('Standard Validation')
        st.write('Calculate the slope and intercept from the standard data which will be used to calculate the concentration.')
        st.write('The result is saved in the selected data folder as an excel file.')
        form = st.form(key='my_form')
        self.sample_type = form.selectbox('Sample Type:', ('Weekly Analysis', 'Chem Div', 'QC Request'))
        self.year = form.text_input('Year (####):')
        self.week = form.text_input('Week (##):')
        self.chemist = form.text_input('Chemist (Aaaa) optional:')
        submit = form.form_submit_button(label='Submit')
        if submit:
            std_data_path = get_std_data_path(self.sample_type, self.year, self.week, self.chemist)
            get_std_info(std_data_path)


def get_std_data_path(sample_type, year, week, chemist):
    """
    Get the paths to the Mnova csv file.
    :param sample_type: Weekly Analysis, Chem Div, or Special request.
    :param year:
    :param week:
    :param chemist: Start with capital case and the rest is all lowercase.
    :return (str): path to the Mnova result csv file.
    """
    std_data_path = std_path + '\\' + year + '\\' + sample_type
    if sample_type == 'Weekly Analysis':
        std_data_path = std_data_path + '\\WK' + week + '\\WK' + week + r' DataAnalysis\WK' + week + r' Mnova\WK' + \
                        week + r' STD\MSQCResults.csv'
    elif sample_type == 'Chem Div':
        std_data_path = std_data_path + '\\CD' + week + '\\CD' + week + r' DataAnalysis\CD' + week + r' Mnova\CD' + \
                        week + r' STD\MSQCResults.csv'
    elif sample_type == 'QC Request':
        std_data_path = std_data_path + '\\' + chemist + '\\WK' + week + chemist + '\\WK' + week + chemist + \
                        r' DataAnalysis\WK' + week + chemist + r' Mnova\WK' + week + chemist + r' STD\MSQCResults.csv'
    else:
        st.write('Input is not right.')
    return std_data_path


def get_std_info(std_data_path):
    """
    Calculate the slope and intercept of the standard data and save it as an excel file in the Mnova STD folder.
    The summary of validation result will show up on the application.
    :return: None
    """
    # Get the weights of chemical compounds.
    std_info = pd.read_excel(std_info_path, sheet_name='std_masses', usecols=[*range(1, 4)], header=0, index_col=0)
    # Get the theoretical standard concentraiton.
    std_calculated_conc = pd.read_excel(std_info_path, sheet_name='std_concentrations', usecols=[*range(0, 2)],
                                        header=0, index_col=0)
    # Get the measured CAD area.
    std_areas = pd.read_csv(std_data_path, index_col=0, header=0)
    # remove some unnecessary values from both data.
    unnecessary_values = ['B04', 'B08', 'G04']
    for idx in std_areas.index:
        if idx[-3:] in unnecessary_values:
            std_areas.drop(idx, axis=0, inplace=True)
    std_calculated_conc.drop([std_calculated_conc.index[7], std_calculated_conc.index[19],
                              std_calculated_conc.index[31]], axis=0, inplace=True)

    # Calculate the intercept and slope.
    lr = linear_model.LinearRegression()

    # As we don't always have a complete dataset, the missing values have to be removed from both measured and
    # calculated concentrations.
    missing_value_idx = []
    conc = []
    for i, c in enumerate(std_areas['Conc (mM)'][0:-3]):
        # Get the indices of missing values.
        if c in ['-', 'not matched']:
            missing_value_idx.append(i)
        # Get the list of measured concentrations.
        else:
            conc.append(float(c))
    cad_area_under_curve = np.array(conc)

    # Convert the calculated concentrations list to np.array.
    theoretical_conc = np.array(std_calculated_conc['Theoretical concentration (ug/mL)'])

    # Delete the missing values from the calculated concentrations.
    theoretical_conc = np.delete(theoretical_conc, missing_value_idx)

    # Process the fitting linear curve with the data.
    lr.fit(theoretical_conc.reshape(len(theoretical_conc), 1), cad_area_under_curve)

    # Get the slope and intercept of the curve.
    slope = float(lr.coef_)
    intercept = float(lr.intercept_)

    # Get the information about the standard validation compound.
    validation_corrected_mass = std_info['Corrected Mass (mg)'][std_info.index[3]]
    validation_molar_mass = std_info['Molar Mass (g/mol)'][std_info.index[3]]

    # The measured mass is resolved in 1 mL as a stock solution.
    # The stock solution is diluted by 10x for the daily use.
    # The daily use sample is diluted by 100x on the plate and measured the concentration.
    calculated_validation_conc = validation_corrected_mass / validation_molar_mass / 1000 * 1000 * 1000
    validation_percent_diffs = []

    for i, c in enumerate(std_areas['Conc (mM)'][-3:]):
        # Check if the concentration is measured or not.
        if c in ['-', 'not matched']:
            pass
        # Use only the numerical data.
        else:
            validation_conc = float(
                (float(c) - intercept) / slope * 100 / std_info['Molar Mass (g/mol)'][std_info.index[3]])
            validation_conc_diff = abs(validation_conc - calculated_validation_conc)
            validation_percent_diff = round(validation_conc_diff / calculated_validation_conc * 100, 2)
            validation_percent_diffs.append(validation_percent_diff)

    # Check if the max percent difference is larger than the threshold.
    if max(validation_percent_diffs) >= threshold_percent_diffs:
        validation = 'NOT VALID'
    else:
        validation = 'VALID'

    # Create the excel file to output the intercept and slope values to the Mnova data folder.
    folder_path = os.path.abspath(os.path.join(std_data_path, os.pardir))

    # Summarize the data for an excel file.
    # Create the list of standard data.
    data_list = [intercept, slope]
    # Create the list of data index.
    data_index = ['Slope', 'Intercept']
    for i, diff in enumerate(validation_percent_diffs):
        data_index.append('Validation ' + str(i + 1) + ' (%)')
        data_list.append(diff)
    # Create the dataframe with the standard data.
    result = pd.DataFrame(data=data_list, index=data_index, columns=['Values'])
    # Save it as an excel file.
    result.to_excel(folder_path + r'\std_info.xlsx')
    # Create the message that summarize the validation result.
    standard_message = 'Missing ' + str(len(missing_value_idx)) + ' out of 36 standard values.'
    validation_message = 'Missing ' + str(3-len(validation_percent_diffs)) + ' out of ' + str(num_val_rep) + \
                         ' validation values.'
    percent_diff_message = '% Difference:  '
    for diff in validation_percent_diffs:
        percent_diff_message = percent_diff_message + str(diff) + ' %, '
    # Send the message.
    st.write(standard_message)
    st.write(validation_message)
    st.write(percent_diff_message)
    st.write('This standard is ' + validation + '.')

