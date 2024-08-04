import streamlit as st
from streamlit_option_menu import option_menu
import home
import prediksi

st.set_page_config(
    page_title="Prediksi Kredit Menggunakan ANN",
)

def main():
    with st.sidebar:
        app = option_menu(
            menu_title='Main Menu',
            options=['Home', 'Prediksi'],
            icons=['house-fill', 'bi-cash'],
            menu_icon='chat-text-fill',
            default_index=0,
            styles={
                "container": {"padding": "5!important"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {"color": "white", "font-size": "14px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )

    if app == "Home":
        home.app()
    elif app == "Prediksi":
        prediksi.app()

if __name__ == "__main__":
    main()
