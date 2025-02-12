import pyreadstat
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import streamlit as st
import seaborn as sns
import tempfile
import os


def preprocess_data_df(data1, data2):
    """
    Preprocess the input dataframes and return the processed DataFrame and unique treatments.

    Parameters:
        df (pd.DataFrame): Main dataset containing treatment data.
        adsl (pd.DataFrame): Dataset containing additional subject-level information.

    Returns:
        tuple: (processed DataFrame, list of unique treatments)
    """
    # Save the uploaded files to temporary files
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".sas7bdat") as temp1, \
    #      tempfile.NamedTemporaryFile(delete=False, suffix=".sas7bdat") as temp2:
        
    #     temp1.write(data1.read())
    #     temp2.write(data2.read())
    #     temp1.flush()
    #     temp2.flush()
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # adrs, meta = pyreadstat.read_sas7bdat(f'C:/Users/jagad/OneDrive/Documents/python/test/{data1}.sas7bdat')
    # adrs, meta = pyreadstat.read_sas7bdat(temp1.name)
    adrs = data1.copy() 
    adrs['USUBJID'] = adrs['USUBJID'].astype(str)
    # df_sub_list = adrs['USUBJID'].unique()

    # print(df_sub_list)

    # adsl, meta = pyreadstat.read_sas7bdat(f'C:/Users/jagad/OneDrive/Documents/python/test/{data2}.sas7bdat')
    # adsl, meta = pyreadstat.read_sas7bdat(temp2.name)
    adsl = data2.copy() 
    adsl['USUBJID'] = adsl['USUBJID'].astype(str)
    # Step 1: Filter for specific PARAMCD
    df = adrs[adrs['PARAMCD'] == 'OVRLRESP']

    # Step 2: Keep relevant columns
    columns_to_keep = ['USUBJID', 'AVISIT', 'AVISITN', 'AVALC', 'ADY', 'TRT01P','PARAMCD']
    df = df[columns_to_keep]



    # Step 3: Calculate the maximum ADY for each subject
    new_data = (
        df.groupby(["USUBJID"], as_index=False)
        .agg(max_ady=("ADY", "max"))
    )
    df = pd.merge(df, new_data, on="USUBJID", how="left", indicator=False)

    # Step 4: Prepare adsl for merging
    adsl_columns = ['USUBJID', 'DTHDT', 'DTHDY', 'DTHFL','TRT01P']
    adsl = adsl[adsl_columns]
    adsl['ADY'] = adsl['DTHDY']
    adsl = adsl[(adsl['DTHFL'] == 'Y') & (adsl['TRT01P'].notna())]
    adsl['AVALC'] = 'Death'
    
    df.to_csv('test1.csv')
    adsl.to_csv('test2.csv')
    # Step 5: Combine adsl with the main dataframe
    df = pd.concat([df, adsl], ignore_index=True)

    adsl_columns2 = ['USUBJID', 'DTHDY']
    adsl2 = adsl[adsl_columns2]
    adsl2 = adsl2.rename(columns={'DTHDY': 'DTHDY2'})

    df = pd.merge(
        df,
        adsl2,  # Select relevant columns from ADSL
        on='USUBJID',
        how='left'
    )

    df.to_csv('test3.csv')
    # Step 6: Forward fill max_ady values
    df['max_ady'] = df.groupby('USUBJID')['max_ady'].ffill()

    # Step 7: Extract numeric portion of USUBJID and sort by USUBJID and ADY
    # df['USUBJID'] = df['USUBJID'].str[2:5]
    df = df.sort_values(by=['USUBJID', 'ADY'], ascending=[True, True])

    # Step 8: Assign numeric codes for USUBJID
    df['USUBJID_numeric'] = pd.Categorical(df['USUBJID']).codes


    # Step 9: Sort and encode treatment groups
    df = df.sort_values(by=['TRT01P'], ascending=True)
    df.to_csv('test4.csv')
    df['TRT01P_numeric'] = pd.Categorical(df['TRT01P']).codes
    df.to_csv('test5.csv')

    # Step 10: Filter out rows with missing max_ady values
    df = df[df['max_ady'].notna()]

    # Step 11: Add a prefix to TRT01P
    df['TRT01P'] = 'TRT-' + df['TRT01P_numeric'].fillna('').astype(str)

    df['DTHDY'] = df['DTHDY'] - df['max_ady']

    # Step 12: Extract unique treatment groups
    unique_records = df[['TRT01P', 'TRT01P_numeric']].drop_duplicates()
    unique_trt = unique_records['TRT01P'].unique()

    return df, list(unique_trt)




def preprocess_data(data1, data2):
    """
    Preprocess the input dataframes and return the processed DataFrame and unique treatments.

    Parameters:
        df (pd.DataFrame): Main dataset containing treatment data.
        adsl (pd.DataFrame): Dataset containing additional subject-level information.

    Returns:
        tuple: (processed DataFrame, list of unique treatments)
    """
        # Save the uploaded files to temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sas7bdat") as temp1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".sas7bdat") as temp2:
        
        temp1.write(data1.read())
        temp2.write(data2.read())
        temp1.flush()
        temp2.flush()
      # st.set_option('deprecation.showPyplotGlobalUse', False)
    # adrs, meta = pyreadstat.read_sas7bdat(f'C:/Users/jagad/OneDrive/Documents/python/test/{data1}.sas7bdat')
    adrs, meta = pyreadstat.read_sas7bdat(temp1.name)
    df_sub_list = adrs['USUBJID'].unique()

    print(df_sub_list)

    isinstance(df_sub_list, list)

    # adsl, meta = pyreadstat.read_sas7bdat(f'C:/Users/jagad/OneDrive/Documents/python/test/{data2}.sas7bdat')
    adsl, meta = pyreadstat.read_sas7bdat(temp2.name)

    # Step 1: Filter for specific PARAMCD
    df = adrs[adrs['PARAMCD'] == 'OVRLRESP']

    # Step 2: Keep relevant columns
    columns_to_keep = ['USUBJID', 'AVISIT', 'AVISITN', 'AVALC', 'ADY', 'TRT01P']
    df = df[columns_to_keep]

    # Step 3: Calculate the maximum ADY for each subject
    new_data = (
        df.groupby(["USUBJID"], as_index=False)
        .agg(max_ady=("ADY", "max"))
    )
    df = pd.merge(df, new_data, on="USUBJID", how="left", indicator=False)

    # Step 4: Prepare adsl for merging
    adsl_columns = ['USUBJID', 'DTHDT', 'DTHDY', 'DTHFL', 'TRT01P']
    adsl = adsl[adsl_columns]
    adsl['ADY'] = adsl['DTHDY']
    adsl = adsl[adsl['DTHFL'] == 'Y']
    adsl['AVALC'] = 'Death'

    # Step 5: Combine adsl with the main dataframe
    df = pd.concat([df, adsl], ignore_index=True)

    adsl_columns2 = ['USUBJID', 'DTHDY']
    adsl2 = adsl[adsl_columns2]
    adsl2 = adsl2.rename(columns={'DTHDY': 'DTHDY2'})

    df = pd.merge(
        df,
        adsl2,  # Select relevant columns from ADSL
        on='USUBJID',
        how='left'
    )

   
    # Step 6: Forward fill max_ady values
    df['max_ady'] = df.groupby('USUBJID')['max_ady'].ffill()

    # Step 7: Extract numeric portion of USUBJID and sort by USUBJID and ADY
    df['USUBJID'] = df['USUBJID'].str.split(r'[\-]').str[3].str[2:5]
    df = df.sort_values(by=['USUBJID', 'ADY'], ascending=[True, True])

    # Step 8: Assign numeric codes for USUBJID
    df['USUBJID_numeric'] = pd.Categorical(df['USUBJID']).codes

    # Step 9: Sort and encode treatment groups
    df = df.sort_values(by=['TRT01P'], ascending=True)
    df['TRT01P_numeric'] = pd.Categorical(df['TRT01P']).codes

    # Step 10: Filter out rows with missing max_ady values
    df = df[df['max_ady'].notna()]

    # Step 11: Add a prefix to TRT01P
    df['TRT01P'] = 'TRT-' + df['TRT01P_numeric'].fillna('').astype(str)

    df['DTHDY'] = df['DTHDY'] - df['max_ady']

    # Step 12: Extract unique treatment groups
    unique_records = df[['TRT01P', 'TRT01P_numeric']].drop_duplicates()
    unique_trt = unique_records['TRT01P'].unique()

    return df, list(unique_trt)

# dfs, unique = preprocess_data('adrs', 'adsl')

# print(dfs[dfs['USUBJID']=='004'])

def plot_avalc_symbols(data, selected_trt, selected_avalc):
    """
    Function to plot a bar chart without markers initially and dynamically add markers
    based on selected AVALC values.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - selected_avalc (list): A list of AVALC values for which symbols will be used in the plot.
    """
    # Define mappings for AVALC to markers and colors
    avalc_markers = {
        'Death': 's',
        'CR': 'p',  # Star filled
        'PR': '^',  # Triangle filled
        'SD': 'o',  # Circle filled
        'PD': 'D',  # Diamond filled
        
    }

    avalc_colors = {
        'CR': '#e377c2',
        'PR': 'blue',
        'SD': 'orange',
        'PD': 'magenta',
        'Death': 'red',
    }

    # Create the bar chart
    plt.figure(figsize=(10, 20))
    # plt.barh(data['USUBJID_numeric'], data['max_ady'], color='skyblue', edgecolor='none', alpha=0.8)
    data = data[data['TRT01P'].isin(selected_trt)]

    # Example dataset (replace with your data's column of treatment groups)
    unique_treatments = data['TRT01P'].unique()

    # Define a set of colors (expand or change as needed)
    color_palette = plt.cm.Pastel1.colors  # Use a colormap (e.g., Pastel1, Set2, etc.)
    num_colors = len(color_palette)

    # Generate the treatment_colors dictionary dynamically
    treatment_colors = {
    treatment: color_palette[i % num_colors]  # Cycle through the color palette if treatments exceed colors
    for i, treatment in enumerate(unique_treatments)
    }

    # Assign colors based on the treatment group
    data['color'] = data['TRT01P'].map(treatment_colors)
    # ylower = data['max_ady'].min()
    xupper = max(data['max_ady'].max(), data['DTHDY2'].max(skipna=True))

    # Dynamically adjust height based on the number of subjects
    num_subjects = data['USUBJID'].nunique()
    height = max(5, num_subjects * 0.5)  # Minimum height of 5, scaling factor of 0.5

    plt.figure(figsize=(10, height))

    for _, row in data.iterrows():
        plt.barh(
            row['USUBJID_numeric'],  # Subject on the y-axis
            row['max_ady'],          # Days of treatment on the x-axis
            color=row['color'],      # Color based on treatment group
            edgecolor='black',
            alpha=0.8
        )
        # Plot "Off Treatment" (dthdy)
        if not pd.isna(row['DTHDY']):
            plt.barh(
                row['USUBJID_numeric'],
                row['DTHDY'],  # Off Treatment duration
                left=row['max_ady'],  # Offset by On Treatment duration
                color='white',
                edgecolor='black',
                alpha=0.6,
                hatch='\\\\',
                label='Off Treatment' if _ == 0 else ""  # Add a single label for the legend
            )

    # Add markers for AVALC at the ADY positions if they are in the selected list
    if selected_avalc:  # Only add markers if AVALC values are selected
        seen_avalc = set()
        for index, row in data.iterrows():
            if row['AVALC'] in selected_avalc:
                marker = avalc_markers.get(row['AVALC'], 'x')  # Default marker if AVALC not found
                color = avalc_colors.get(row['AVALC'], 'black')  # Default color if AVALC not found
                label = row['AVALC'] if row['AVALC'] not in seen_avalc else None  # Add label only once
                plt.plot(row['ADY'], row['USUBJID_numeric'], marker=marker, color=color, markersize=6, label=label)
                seen_avalc.add(row['AVALC'])

        # Create custom legend with only the selected AVALC values
        legend_elements1 = [
            Line2D([0], [0], marker=avalc_markers[avalc], color='w', markerfacecolor=avalc_colors[avalc], markersize=8, label=avalc)
            for avalc in selected_avalc
        ]
        
        legend_elements2 = [
        Line2D([0], [0], color=color, lw=4, label=treatment)
            for treatment, color in treatment_colors.items()
        ]

        legend_elements2.append(
        Patch(facecolor='white', edgecolor='black', hatch='\\\\', label='Off Treatment')
        )
    

    # Update the y-axis ticks to show original USUBJID values
    plt.yticks(data['USUBJID_numeric'], data['USUBJID'])

    # Add labels and title
    plt.xlabel('Days of Treatment')
    plt.ylabel('Subjects')
    # plt.title('Swimmer Plot for Treatment Exposure')
    plt.margins(x=0, y=0.01)
    ax = plt.gca()
    legend1 = ax.legend(handles=legend_elements1, title='Response', loc='lower right')
    ax.legend(handles=legend_elements2, title='Treatments', loc='upper right')

    ax.add_artist(legend1) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, xupper+100) 

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())


def plot_(data,selected_trt):
    """
    Function to plot a bar chart without markers initially and dynamically add markers
    based on selected AVALC values.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - selected_avalc (list): A list of AVALC values for which symbols will be used in the plot.
    """
    # Create the bar chart
    

    data = data[data['TRT01P'].isin(selected_trt)]

    # Example dataset (replace with your data's column of treatment groups)
    unique_treatments = data['TRT01P'].unique()

    # Define a set of colors (expand or change as needed)
    color_palette = plt.cm.Pastel1.colors  # Use a colormap (e.g., Pastel1, Set2, etc.)
    num_colors = len(color_palette)

    # Generate the treatment_colors dictionary dynamically
    treatment_colors = {
    treatment: color_palette[i % num_colors]  # Cycle through the color palette if treatments exceed colors
    for i, treatment in enumerate(unique_treatments)
    }
  
    # Assign colors based on the treatment group
    data['color'] = data['TRT01P'].map(treatment_colors)
    xupper = max(data['max_ady'].max(), data['DTHDY2'].max(skipna=True))

    # Dynamically adjust height based on the number of subjects
    num_subjects = data['USUBJID'].nunique()
    height = max(5, num_subjects * 0.5)  # Minimum height of 5, scaling factor of 0.5

    plt.figure(figsize=(10, height))
    # for _, row in data.iterrows():
    for _, row in data.iterrows():
        # Plot "On Treatment" (max_ady)
        plt.barh(
            row['USUBJID_numeric'],
            row['max_ady'],  # On Treatment duration
            color=row['color'],
            edgecolor='black',
            alpha=0.8,
            label='On Treatment' if _ == 0 else ""  # Add a single label for the legend
        )

        # Plot "Off Treatment" (dthdy)
        if not pd.isna(row['DTHDY']):
            plt.barh(
                row['USUBJID_numeric'],
                row['DTHDY'],  # Off Treatment duration
                left=row['max_ady'],  # Offset by On Treatment duration
                color='white',
                edgecolor='black',
                alpha=0.6,
                hatch='\\\\',
                label='Off Treatment' if _ == 0 else ""  # Add a single label for the legend
            )

    legend_elements = [
        Line2D([0], [0], color=color, lw=4, label=treatment)
        for treatment, color in treatment_colors.items()
    ]
    
    legend_elements.append(
        Patch(facecolor='white', edgecolor='black', hatch='\\\\', label='Off Treatment')
    )
    
    plt.legend(handles=legend_elements, title='Treatments', loc='best')
    # Update the y-axis ticks to show original USUBJID values
    plt.yticks(data['USUBJID_numeric'], data['USUBJID'])

    # Add labels and title
    plt.xlabel('Days of Treatment')
    plt.ylabel('Subjects')
    # plt.title('Swimmer Plot for Treatment Exposure')
    # Adjust axis limits to start at (0, 0)
    plt.margins(x=0, y=0.01)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, xupper+100) 

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())


data_source = st.sidebar.radio("Select Data Source", ["Pre-loaded CSVs", "Upload CSVs"])

st.sidebar.markdown("## User Inputs")
st.sidebar.markdown("\n")
upload = st.sidebar.file_uploader('Choose a file', accept_multiple_files=True)
st.sidebar.markdown("\n ## Responses")

if data_source == "Pre-loaded CSVs":
    # Directory containing pre-loaded CSV files
    csv_folder = "./csv_folder"
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    st.markdown('#### Swimmer Plot for Treatment Exposure and Objective Response')
    # Sidebar: Select CSV file
    st.sidebar.markdown("## Select a CSV File")
    selected_csv1 = st.sidebar.selectbox("Choose first file", csv_files, index=0)
    selected_csv2 = st.sidebar.selectbox("Choose second file", csv_files, index=1)

    # Load the selected CSV files
    file_path1 = os.path.join(csv_folder, selected_csv1)
    file_path2 = os.path.join(csv_folder, selected_csv2)

    upload1 = pd.read_csv(file_path1)
    upload2 = pd.read_csv(file_path2)

    processed_df, unique_trt = preprocess_data_df(upload1, upload2)

    

    selected_trt = st.sidebar.multiselect("Select Treatment to Display", options=unique_trt, default=unique_trt)
    # Filter the data based on selected treatments
    df2 = processed_df[processed_df['TRT01P'].isin(selected_trt)]

    # Recompute USUBJID_numeric dynamically based on the filtered data
    df2['USUBJID_numeric'] = pd.Categorical(df2['USUBJID']).codes

    # Pass the updated DataFrame to the plot functions
    selected_trt = df2['TRT01P'].unique()
    selected_avalc = st.sidebar.multiselect("Select Responses to Highlight", options=['CR', 'PR', 'SD', 'PD', 'Death'], default=[])
    # Add feedback if no AVALC values are selected
    if selected_avalc:
        with st.spinner("Generating plot with symbols..."):
                
            plot_avalc_symbols(df2, selected_trt, selected_avalc)
    else:
        st.warning("No Response values selected. Plotting without symbols.")
        with st.spinner("Generating plot..."):
                
            plot_(df2,selected_trt)

elif data_source == "Upload CSVs":

    # st.write("Selected AVALC values:", selected_avalc)
    with st.container():
        if upload is not None and len(upload) == 2:
            st.markdown('#### Swimmer Plot for Treatment Exposure and Objective Response')
            # upload_files = [file.name.replace('.sas7bdat', '') for file in upload]
                
            upload1 = upload[0]
            upload2 = upload[1]
            processed_df, unique_trt = preprocess_data(upload1, upload2)
            selected_trt = st.sidebar.multiselect("Select Treatment to Display", options=unique_trt, default=unique_trt)
            # Filter the data based on selected treatments
            df2 = processed_df[processed_df['TRT01P'].isin(selected_trt)]

            # Recompute USUBJID_numeric dynamically based on the filtered data
            df2['USUBJID_numeric'] = pd.Categorical(df2['USUBJID']).codes

            # Pass the updated DataFrame to the plot functions
            selected_trt = df2['TRT01P'].unique()
            selected_avalc = st.sidebar.multiselect("Select Responses to Highlight", options=['CR', 'PR', 'SD', 'PD', 'Death'], default=[])
            # Add feedback if no AVALC values are selected
            if selected_avalc:
                with st.spinner("Generating plot with symbols..."):
                        
                    plot_avalc_symbols(df2, selected_trt, selected_avalc)
            else:
                st.warning("No Response values selected. Plotting without symbols.")
                with st.spinner("Generating plot..."):
                        
                    plot_(df2,selected_trt)
        else:
            st.info("Please upload a file to generate the plot.")


