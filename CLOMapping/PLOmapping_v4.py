# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 07:57:52 2023

@author: Warong
"""

###################################################
### *****************************************
### USE auto-py-to-exe  to create EXE files
###  RUN "auto-py-to-exe" in ANACONDA prompt
#############################################
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# EXECUTABLE PATH HANDLING ---
# This function ensures your script looks for @data and @output
# exactly next to where the .exe file is located.
def get_base_path():
    if getattr(sys, 'frozen', False):
        # If it's running as an EXE
        return os.path.dirname(sys.executable)
    else:
        # If it's running as a normal Python script
        return os.path.dirname(os.path.abspath(__file__))
        
BASE_DIR = get_base_path()
DATA_DIR = os.path.join(BASE_DIR, '@data')
OUTPUT_DIR = os.path.join(BASE_DIR, '@output')

# Setup paths based on the dynamic base directory
clomap_dir = os.path.join(DATA_DIR, 'clomap')
plomap_dir = os.path.join(DATA_DIR, 'plomap')

# Check if folders exist to prevent crashing
if not os.path.exists(clomap_dir) or not os.path.exists(plomap_dir):
    exit_program("ERROR: Could not find @data/clomap or @data/plomap folders.")

# Get list of files from clomap and plomap folders
clo_files = os.listdir(clomap_dir)
plo_files = os.listdir(plomap_dir)

# Use a list comprehension to get the left subset for each string
clo_check = [s[3:] for s in clo_files]
plo_check = [s[3:] for s in plo_files]
if clo_check != plo_check:
    print('ERROR: Files in clomap and plomap folders are not matched')
    input("Press Enter to exit...")
    sys.exit()
else:
    print('Files are matched')

# Setup output folder safely
student_out_dir = os.path.join(OUTPUT_DIR, 'student')
if not os.path.exists(student_out_dir):
    os.makedirs(student_out_dir)

# FIXED PATH
credit_data = pd.read_csv(os.path.join(DATA_DIR, 'credits.csv'))
credit_names = credit_data['courses'].values.astype(str)

###########################################################################################################
#%% FUNCTIONssss

# function to match course score to CLO
def score_to_clo(clo_file):
    ## get CLO map data (2nd column and beyond)
    data = np.genfromtxt('@data/clomap/'+clo_file, delimiter=',', skip_header=1)[:,1:]
    score = data[:,0]
    # subset clo map and set all nan values to zeros
    clo_map = np.nan_to_num(data[:,1:])
    # calculate CLO weight for each activity
    clo_weight = clo_map / clo_map.sum(axis=1, keepdims=True)
    # Calculate overall score contribution to each CLO
    out_score_to_clo = np.dot(score,clo_weight)
    return out_score_to_clo

# Mapping CLO scores to PLOs
# weight by "credit hour"
def score_to_plo(clo_file):
    score_clo = score_to_clo(clo_file) # get array as the results
    course_name = clo_file[3:clo_file.find('.')] # get course name from file name (the 4th letter to the .)
    credit = credit_data.iloc[np.argmax(credit_names == course_name),1] # get credit of the course from credit_data using course name index
    # get plo mapping file the same name as clo files
    plo_file = 'p' + clo_file[1:]
    plo_map = np.nan_to_num(np.genfromtxt('@data/plomap/'+plo_file, delimiter=',', skip_header=1)[:,1:])
    # Calculate shared weights of each PLO mapped from CLO
    plo_weight = plo_map / plo_map.sum(axis=1, keepdims=True)
    score_to_plo = np.dot(score_clo, plo_weight)
    # Multiply PLO score with credit
    out_score_to_plo = score_to_plo * credit
    return out_score_to_plo

# Function to convert a letter to its corresponding number
def letter_to_number(letter):
    if letter.isalpha():
        # Convert uppercase letters to numbers (A=1, B=2, ..., Z=26)
        return ord(letter.upper()) - ord('A') + 1
    else:
        # Leave non-alphabetic characters unchanged
        return letter

# Function to map letter grades to numerical scores
# Use dictionary instead of if else
# --- NEW: LOAD GRADE MAP FROM CSV ---
grade_map_path = os.path.join(DATA_DIR, 'grade_map.csv')
if not os.path.exists(grade_map_path):
    print('ERROR: Could not find grade_map.csv in the @data folder.')
    input("Press Enter to exit...")
    sys.exit()
# Read the CSV file
grade_map_data = pd.read_csv(grade_map_path)
# Convert the first two columns into a dictionary
# We force the grades (keys) to be uppercase strings with no spaces for safe matching
# We force the scores (values) to be floats
GRADE_MAP = {
    str(k).strip().upper(): float(v) 
    for k, v in zip(grade_map_data.iloc[:, 0], grade_map_data.iloc[:, 1])
}
# SAFER GRADE MAPPING ---
def map_grades(grade):
    # Check if the grade is a string before stripping (in case of NaNs/floats)
    if isinstance(grade, str):
        grade = grade.strip().upper()
    # Use the globally loaded dictionary instead of the hardcoded one
    # Default to 0.0 if the grade isn't found in the CSV
    return GRADE_MAP.get(grade, 0.0)

#%% Running the app to calculate course-wise PLO score
# Loop through file
stacked_list = []
for i in clo_files:
    print(i)
    out = score_to_plo(i)
    stacked_list.append(out)
# convert to multiple-row array
plo_stack = np.vstack(stacked_list)
# calculate courses contribution to PLO
plo_cont = (plo_stack / plo_stack.sum(axis=0)) * 100
# calculate totals and weights of each PLO
total = plo_stack.sum(axis=0)
weight = (total / total.sum()) * 100
plo_stack = np.vstack([stacked_list, total, weight])
# Convert NumPy array to Pandas DataFrame
# df for PLO score
plo_df = pd.DataFrame(plo_stack)
# df for PLO contribution
plo_cont_df = pd.DataFrame(plo_cont)

# Set row names
row_names = [name[3:name.find('.')] for name in clo_files]
# set row names for contribution matrix
plo_cont_df.set_index(pd.Index(row_names), inplace=True)
# set row name to overall PLO table + total row + weight row
row_names += ['Total','Weight']
plo_df.set_index(pd.Index(row_names), inplace=True)

# Set column names
plo_df.columns = ['PLO1','PLO2','PLO3','PLO4','PLO5']
plo_cont_df.columns = ['PLO1','PLO2','PLO3','PLO4','PLO5']

# Add new calculation
plo_df_output = plo_df.copy() # "plo_df_output = plo_df" do not give the new copy but the new reference
plo_df_output['Sum'] = plo_df_output.sum(axis=1) # Add a new column 'Sum' that represents the sum across each row, to see the total credit point of each course
plo_cont_df['average'] = plo_cont_df.mean(axis=1) # average over column
plo_cont_df.loc['average'] = plo_cont_df.mean(axis=0) # Add a new row 'average' with the sum of each column for comparing the contribution

# Specify the file path for saving the CSV file
# Save the DataFrame to a CSV file
plo_df_output.to_csv('@output/plo_scores.csv')

###############################################################################################################
print("\nGenerating PLO Weights Bar Plot...")
# 1. Read the saved CSV file back in (setting the first column as the index)
plo_df_saved = pd.read_csv('@output/plo_scores.csv', index_col=0)
# 2. Extract the 'Weight' row and drop the 'Sum' column
# errors='ignore' ensures it won't crash if 'Sum' happens to be missing
plo_weights = plo_df_saved.loc['Weight'].drop('Sum', errors='ignore')
# 3. Setup the plot
plt.figure(figsize=(8, 6)) # Adjust width and height
# Plotting the bar chart with some nice colors
bars = plo_weights.plot(
    kind='bar', 
    color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'],
    edgecolor='black'
)
# 4. Add titles and labels
plt.title('Overall Weight Distribution per PLO', fontsize=14, fontweight='bold')
plt.xlabel('Program Learning Outcomes (PLO)', fontsize=12)
plt.ylabel('Weight (%)', fontsize=12)
# Keep the X-axis labels horizontal
plt.xticks(rotation=0)
# Add a little headroom above the tallest bar for the text labels
plt.ylim(0, plo_weights.max() + 5)
# 5. Add exact percentage values on top of each bar for clarity
for i, value in enumerate(plo_weights):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha='center', va='bottom', fontweight='bold')
# 6. Save the plot to the output folder
plt.tight_layout()
barplot_path = '@output/plo_scores.jpg'
plt.savefig(barplot_path, dpi=150)
plt.close() # Close to free up memory
##############################################################################################################

# export contribution matrix
plo_cont_df.to_csv('@output/plo_contributions.csv')

#%% STUDENT ANALYSIS
print('\nStudent Analysis')

# SAFE PATHING
student_data_dir = os.path.join(DATA_DIR, 'student')

if not os.path.exists(student_data_dir):
    print('ERROR: No @data/student folder found.')
    time.sleep(3)
    sys.exit()

student_files = os.listdir(student_data_dir)
if not student_files:
    print('No student files found in directory.')
    time.sleep(3)
    sys.exit()

for file in student_files:
    print(f"Processing: {file}")
    
    # SAFE PATHING
    student_df = pd.read_csv(os.path.join(student_data_dir, file))
    student_id = student_df.iloc[:,0]
    grade_df = student_df.iloc[:,1:].map(map_grades) 
    
    students_plo_score = []
    students_plo_percent = []
    
    for i in range(len(grade_df)): 
        df_temp = grade_df.iloc[i,:] 
        courses = grade_df.columns[df_temp.notna()]
        scores = df_temp[df_temp.notna()]
        selected_plo = plo_df.loc[courses]
        total_selected_plo = selected_plo.sum() 
        
        plo_score = pd.Series(np.dot(scores, selected_plo))
        plo_score.index = ['PLO1', 'PLO2', 'PLO3', 'PLO4', 'PLO5']
        students_plo_score.append(plo_score)
        
        if i == 0:  
            plo_percent = (plo_score / total_selected_plo) * 100
            ref_plo_score = plo_score.copy()
        else:       
            plo_percent = (plo_score / ref_plo_score) * 100
        
        students_plo_percent.append(plo_percent)

    students_plo_score_df = pd.DataFrame(students_plo_score)
    students_plo_score_df.index = student_id
    students_plo_percent_df = pd.DataFrame(students_plo_percent)
    students_plo_percent_df.index = student_id

    # SAFE PATHING FOR EXPORT
    students_plo_score_df.to_csv(os.path.join(student_out_dir, f'score_{file}'))
    students_plo_percent_df.to_csv(os.path.join(student_out_dir, f'percent_{file}'))

#%% Plotting Radar Plot for each student
plt.ioff()  # Turn off interactive mode
df = students_plo_percent_df.copy()

print("\nPlotting Radar Plots...")
time.sleep(1)

# --- OPTIMIZATION: Calculate static math OUTSIDE the loop ---
categories = list(df)[0:] 
N = len(categories)
base_angles = [n / float(N) * 2 * pi for n in range(N)]
base_angles += base_angles[:1] 
rotation_angle = 55 * pi / 180  
# Pre-calculate the rotated and reversed angles for all plots
final_angles = [(angle + rotation_angle) for angle in base_angles][::-1]
circle_angles = np.linspace(0, 2 * np.pi, 200) 
# -----------------------------------------------------------

for i in range(len(df)):
    student_id = df.index[i]
    
    values = df.iloc[i,:].values.flatten().tolist()
    values += values[:1]
    
    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    plt.xticks(final_angles[:-1], categories, color='black', size=12)
    ax.set_rlabel_position(45)
    ax.grid(color='grey')
    
    if i > 0:
        # Plot reference circle
        ax.plot(circle_angles, [50] * len(circle_angles), color='red', linestyle='-', linewidth=1)
    
    plt.yticks([25,50,75,100], ["25%","50%","75%","100%"], color="black", size=8)
    plt.ylim(0,100)
    
    if i == 0: 
        ax.plot(final_angles, values, linewidth=2, linestyle='solid', color="red")
        ax.fill(final_angles, values, 'red', alpha=0.25)
        plt.title('The Reference Scores \n *Percentage calculated based on all available courses*')
    else: 
        ax.plot(final_angles, values, linewidth=1.5, linestyle='solid', color="green")
        ax.fill(final_angles, values, 'green', alpha=0.1)
        plt.title(f'Student: {student_id}')
    
    plt.tight_layout()
    
    # SAFE PATHING FOR IMAGE SAVE
    save_path = os.path.join(student_out_dir, f'student_{student_id}.jpg')
    plt.savefig(save_path, dpi=150)
    
    # OPTIMIZATION: Completely clear and close the figure from memory
    plt.clf()
    plt.close(fig) 

print("\nProcess Complete.")
time.sleep(1)
input("Press Enter to exit...")