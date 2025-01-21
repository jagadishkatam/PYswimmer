import pyreadstat
import pandas as pd
import datetime as dt
import numpy as np
import pyreadstat
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st
import streamlit.web.bootstrap
from streamlit.web import cli as stcli
import sys
import random
import seaborn as sns


def preprocess_data(data1, data2):
    """
    Preprocess the input dataframes and return the processed DataFrame and unique treatments.

    Parameters:
        df (pd.DataFrame): Main dataset containing treatment data.
        adsl (pd.DataFrame): Dataset containing additional subject-level information.

    Returns:
        tuple: (processed DataFrame, list of unique treatments)
    """
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    adrs, meta = pyreadstat.read_sas7bdat(f'C:/Users/jagad/OneDrive/Documents/python/test/{data1}.sas7bdat')

    df_sub_list = adrs['USUBJID'].unique()

    print(df_sub_list)

    isinstance(df_sub_list, list)

    adsl, meta = pyreadstat.read_sas7bdat(f'C:/Users/jagad/OneDrive/Documents/python/test/{data2}.sas7bdat')

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

dfs, unique = preprocess_data('adrs', 'adsl')

print(dfs[dfs['USUBJID']=='004'])

def plot_avalc_symbols(data, selected_trt, selected_avalc):
    """
    Function to plot a bar chart without markers initially and dynamically add markers
    based on selected AVALC values.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - selected_avalc (list): A list of AVALC values for which symbols will be used in the plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import streamlit as st

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
    xupper = data['max_ady'].max()

    # Create the bar chart
    plt.figure(figsize=(10, 7))

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
        Line2D([0], [0], color='white', lw=4, label='Off Treatment')
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
    xupper = max(data['max_ady'].max(), data['DTHDY'].max(skipna=True))

    plt.figure(figsize=(10, 7))
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
                label='Off Treatment' if _ == 0 else ""  # Add a single label for the legend
            )

    legend_elements = [
        Line2D([0], [0], color=color, lw=4, label=treatment)
        for treatment, color in treatment_colors.items()
    ]
    
    legend_elements.append(
        Line2D([0], [0], color='white', lw=4, label='Off Treatment')
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


st.sidebar.markdown("## User Inputs")
st.sidebar.markdown("\n")
upload = st.sidebar.file_uploader('Choose a file')
st.sidebar.markdown("\n ## Responses")

# st.write("Selected AVALC values:", selected_avalc)
with st.container():
    if upload is not None:
        st.subheader('Swimmer Plot for Treatment Exposure and Objective Response')
        upload1 = upload.name.replace('.sas7bdat','')
        processed_df, unique_trt = preprocess_data(upload1, "adsl")
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
