import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def app():
    st.subheader('Exploratory Data Analysis')
    # st.markdown('Studi Kasus: [PNM Mekaar]')
    uploaded_file = st.file_uploader("")
    if uploaded_file == None:
        # dataset_file = "dataset/Dataset PNM Mekaar.csv"
        dataset_df = pd.read_csv("https://github.com/LahuddinLubis/deploy-model-ann-prediksi-kredit/blob/master/Dataset%20PNM%20Mekaar.csv")
        st.write(dataset_df)
    else:
        dataset_df = pd.read_csv(uploaded_file)
        st.write(dataset_df)

    # add tabs to the main panel of the app
    tab1, tab2, tab3 = st.tabs(["Dataset Info", "Numeric Features", "Categorical Features"])
    
    # Display Dataset Information
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # extract columns type from the uploaded dataset
            st.markdown("Tipe Data Pada Dataset")
            # get feature names
            columns = list(dataset_df.columns)
            # create dataframe
            column_info_table = pd.DataFrame({
                "Kolom": columns,
                "Tipe Data": dataset_df.dtypes.tolist()
            })     
            # display pandas dataframe as a table
            st.dataframe(column_info_table) # st.dataframe(column_info_table, hide_index=True)
        with col2:
            # extract meta-data from the uploaded dataset
            st.markdown("Informasi Dataset")
            row_count = dataset_df.shape[0]
            column_count = dataset_df.shape[1]
         
            # Use the duplicated() function to identify duplicate rows
            duplicates = dataset_df[dataset_df.duplicated()]
            duplicate_row_count =  duplicates.shape[0]
            missing_value_row_count = dataset_df[dataset_df.isna().any(axis=1)].shape[0]
     
            table_markdown = f"""
                | Deskripsi | Value | 
                |---|---|
                | Jumlah Baris | {row_count} |
                | Jumlah Kolom | {column_count} |
                | Jumlah Baris yang Duplikat | {duplicate_row_count} |
                | Jumlah Baris yang Missing Values | {missing_value_row_count} |
                """
            st.markdown(table_markdown)            
    
    # Display numeric features related information
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            # find numeric features  in the dataframe
            numeric_cols = dataset_df.select_dtypes(include='number').columns.tolist()
            # add selection-box widget
            selected_num_col = st.selectbox("Pilih kolom yang ingin di explore?", numeric_cols)

            # add column statistics table
            st.write("Kolom yang dipilih:", selected_num_col)    
            col_info = {}
            col_info["Number of Unique Values"] = len(dataset_df[selected_num_col].unique())
            col_info["Number of Rows with Missing Values"] = dataset_df[selected_num_col].isnull().sum()
            col_info["Number of Rows with 0"] = dataset_df[selected_num_col].eq(0).sum()
            col_info["Number of Rows with Negative Values"] = dataset_df[selected_num_col].lt(0).sum()
            col_info["Average Value"] = dataset_df[selected_num_col].mean()
            # col_info["Standard Deviation Value"] = dataset_df[selected_num_col].std()
            col_info["Minimum Value"] = dataset_df[selected_num_col].min()
            col_info["Maximum Value"] = dataset_df[selected_num_col].max()
            col_info["Median Value"] = dataset_df[selected_num_col].median()
     
            info_df = pd.DataFrame(list(col_info.items()), columns=['Deskripsi', 'Value']) 
            # display dataframe as a markdown table
            st.dataframe(info_df)
        with col2:
            # add a histogram chart for numeric feature
            st.write("Histogram dari ", selected_num_col)
            fig = px.histogram(dataset_df, x=selected_num_col)
            st.plotly_chart(fig, use_container_width=True)

    # Display categorical features related information
    with tab3:
        # find categorical columns in the dataframe
        cat_cols = dataset_df.select_dtypes(include='object')
        cat_cols_names = cat_cols.columns.tolist()
        # add select widget
        selected_cat_col = st.selectbox("Plih kolom yang ingin di explore?", cat_cols_names)
         
        # add categorical column statistics table
        st.write("Kolom yang dipilih:", selected_cat_col) 
        cat_col_info = {}
        cat_col_info["Number of Unique Values"] = len(dataset_df[selected_cat_col].unique())
        cat_col_info["Number of Rows with Missing Values"] = dataset_df[selected_cat_col].isnull().sum()
        cat_col_info["Number of Empty Rows"] = dataset_df[selected_cat_col].eq("").sum()
        cat_col_info["Number of Rows with Only Whitespace"] = len(dataset_df[selected_cat_col][dataset_df[selected_cat_col].str.isspace()])
        cat_col_info["Number of Rows with Only Lowercases"] = len(dataset_df[selected_cat_col][dataset_df[selected_cat_col].str.islower()])
        cat_col_info["Number of Rows with Only Uppercases"] = len(dataset_df[selected_cat_col][dataset_df[selected_cat_col].str.isupper()])
        cat_col_info["Number of Rows with Only Alphabet"] = len(dataset_df[selected_cat_col][dataset_df[selected_cat_col].str.isalpha()])
        cat_col_info["Number of Rows with Only Digits"] = len(dataset_df[selected_cat_col][dataset_df[selected_cat_col].str.isdigit()])
        cat_col_info["Mode Value"] = dataset_df[selected_cat_col].mode()[0]
 
        cat_info_df = pd.DataFrame(list(cat_col_info.items()), columns=['Deskripsi', 'Value'])
        st.dataframe(cat_info_df)
