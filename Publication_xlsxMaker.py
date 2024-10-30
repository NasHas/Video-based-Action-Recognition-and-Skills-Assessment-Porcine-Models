'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'### The following script is made as part of the study #Video-based robotic surgical action recognition and skills assessment on porcine models using deep learning###'
'### The code is open-source. However, when using the code, please make a reference to our paper and repository.""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'


import os
import pandas as pd

# Get the current working directory
folder_path = r'FOLDER PATH'

# Get a list of TSV files in the folder
tsv_files = [file for file in os.listdir(folder_path) if file.endswith('.tsv')]

# Loop through each TSV file and convert to XLSX
for file in tsv_files:
    # Read the TSV file using pandas
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, sep='\t')

    # Generate the output XLSX file name
    output_file = os.path.splitext(file)[0] + '.xlsx'
    output_path = os.path.join(folder_path, output_file)

    # Write the DataFrame to an XLSX file
    df.to_excel(output_path, index=False)