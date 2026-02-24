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

clo_files = os.listdir('@data/clomap')
plo_files = os.listdir('@data/plomap')

# Use a list comprehension to get the left subset for each string
clo_check = [s[3:] for s in clo_files]
plo_check = [s[3:] for s in plo_files]
if clo_check != plo_check:
    print('ERROR: Files in clomap and plomap folders are not matched')
    sys.exit()
else:
    print('Files are matched')

# Check if the folder exists
if not os.path.exists('@output/student'):
    # If not, create it
    os.makedirs('@output/student')

credit_data = pd.read_csv('@data/credits/credits.csv')
credit_names = credit_data['courses'].values.astype(str)

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
def map_grades(grade):
    grade_map = {'A': 1.0, 'B+': 0.75, 'B': 0.5, 'C+': 0.25, 'C': 0.0, 'D+': 0.0, 'D': 0.0, 'F': 0.0}
    return grade_map.get(grade, 0.0)  # Default to 0.0 for unknown grades

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

# export contribution matrix
plo_cont_df.to_csv('@output/plo_contributions.csv')

#%% STUDENT ANALYSIS
print('\nStudent Analysis')
student_files = os.listdir('@data/student')
if not student_files:
    print('No student file')
    # Sleep for 3 seconds
    time.sleep(3)
    sys.exit()

for file in student_files:
    # Calculate scores for each student
    # Read a CSV file into a DataFrame
    print(file)
    student_df = pd.read_csv('@data/student/'+file)
    # Student ID
    student_id = student_df.iloc[:,0]
    # Apply the function to the 'Grades' column
    grade_df = student_df.iloc[:,1:].map(map_grades) ## can be substituted with grade_df = student_df.iloc[:,1:].map(map_grades)
    ###grade_df = student_df.iloc[:,1:].map(map_grades)
    # Create blank list
    students_plo_score = []
    students_plo_percent = []
    # Loop through each student to calculate score
    for i in range(len(grade_df)): # Looping through student i
        # Select grade score of i student
        df_temp = grade_df.iloc[i,:] # get percent score (proportion of CLO or PLO student will get) for each course
        
        # get the name (or id) of courses that student i has grade (not nan data)
        # for reference student this is the course that is minimum required to graduate the program
        courses = grade_df.columns[df_temp.notna()]
        
        # get grading score of courses that student i has grade (not nan data)
        scores = df_temp[df_temp.notna()]
        
        # Select rows from plo weight score based on the name (id) of the courses
        selected_plo = plo_df.loc[courses]
        
        # sum scores over row (produce sum of PLO scores over all courses for student i)
        total_selected_plo = selected_plo.sum() 
        
        # dot product of student i grading score (proportion) from each course with plo scores for each course give score of student i get from each PLO
        plo_score = pd.Series(np.dot(scores, selected_plo))
        
        # set index names
        plo_score.index = ['PLO1', 'PLO2', 'PLO3', 'PLO4', 'PLO5']
        
        # stack current student score to the previous loops
        students_plo_score.append(plo_score)
        
        # calculate PLO percent student get from scores
        # The PLO percent of student i = (PLO scores of stuent i) / (PLO scores of reference student)
        if i == 0:  # If student is the reference (1st row)
            plo_percent = (plo_score / total_selected_plo) * 100
            ref_plo_score = plo_score.copy()
        else:       # If the student is from 2nd row and on and on... Calculate percent based on reference student (1st row)
            plo_percent = (plo_score / ref_plo_score) * 100
        # append data to dataframe
        students_plo_percent.append(plo_percent)

    # Convert to df and export csv
    students_plo_score_df = pd.DataFrame(students_plo_score)
    students_plo_score_df.index = student_id
    students_plo_percent_df = pd.DataFrame(students_plo_percent)
    students_plo_percent_df.index = student_id

    # Specify the file path for saving the CSV file
    file_dir = '@output/student/'
    # Save the DataFrame to a CSV file
    students_plo_score_df.to_csv(file_dir+'score_'+file)
    students_plo_percent_df.to_csv(file_dir+'percent_'+file)

#%% Experiments
# Libraries
import matplotlib.pyplot as plt
from math import pi

### Plotting radar plot or spider plot for every students
 
# block # df = students_plo_score_df.copy()  # students_plo_score_df  or students_plo_percent_df
# Calculate percent compare to maximu score of each PLO
# block # max_overall = df.iloc[0]
# block # df = (df/max_overall) * 100

plt.ioff()  # Turn off interactive mode
df = students_plo_percent_df.copy()

print("\nPlotting Radar Plot")
time.sleep(1)
for i in range(len(df)):
    # get student id
    student_id = df.index[i]
    # number of variable
    categories=list(df)[0:] # select student
    N = len(categories)
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values=df.iloc[i,:].values.flatten().tolist()
    values += values[:1]
    values
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] 
    # Rotate the start angle by "dg" degrees (convert to radians)
    dg = 55
    rotation_angle = dg * pi / 180  
    angles = [(angle + rotation_angle) for angle in angles]
    # Reverse the angles to make the plot go clockwise
    angles = angles[::-1] 
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='black', size=12)
    # Draw ylabels
    ax.set_rlabel_position(45)
    # color of grid
    ax.grid(color='grey')
    
    # Plot red line at 50% to be the reference line (easier visualize)
    if i > 0:
        # Add a circular line at a specific radius
        circle_radius = 50  # Change this value to adjust the circle's radius
        circle_angles = np.linspace(0, 2 * np.pi, 200)  # More points for a smoother circle
        ax.plot(circle_angles, [circle_radius] * len(circle_angles), color='red', linestyle='-', linewidth=1)
    
    # plot grid label
    plt.yticks([25,50,75,100], ["25%","50%","75%","100%"], color="black", size=8)
    plt.ylim(0,100)
    # Plot data
    if i == 0: # red color for reference plot
        ax.plot(angles, values, linewidth=2, linestyle='solid', color="red")
        # Fill area
        ax.fill(angles, values, 'red', alpha=0.25)
    else: # green color for real-data plot
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', color="green")
        # Fill area
        ax.fill(angles, values, 'green', alpha=0.1)
    # Add a title to the plot
    plt.title('Student: '+str(student_id))
    # Change title for reference student
    if i == 0:
        plt.title('The Reference Scores \n *Percentage calculated based on all available courses*')
    
    # Tight layout
    plt.tight_layout()
    # Save the plot as a JPG file
    plt.savefig('@output/student/student'+str(student_id)+'.jpg', dpi=150)
    plt.close()  # Close the figure to release resources
    plt.figure()  # Use plt.figure instead of plt.show if do not want to show th plot

time.sleep(1)
input("Press Enter to exit...")