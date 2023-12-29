import sys

sys.path.append("../../Data-Mining-Project")

import streamlit as st
from frame1 import frame1
from frame2 import frame2

st.set_page_config(layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    tab1, tab2 = st.tabs(["Frame 1", "Frame 2"])

    with tab1:
        frame1()

    with tab2:
        frame2()


if __name__ == "__main__":
    main()
