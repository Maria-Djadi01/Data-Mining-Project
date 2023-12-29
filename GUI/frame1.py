# frame_module.py

import streamlit as st


def frame1():
    # tab1 = st.container(border=True)

    st.write("This is inside the container")

    st.sidebar.write("This is inside the sidebar")

    col1, col2 = st.columns([1, 2])

    with col1:
        num_buttons = 5

        # Create buttons dynamically using a loop
        button_states = [st.button(f"Button {i + 1}") for i in range(num_buttons)]

        match button_states:
            case [True, False, False, False, False]:
                col2.write("Button 1 clicked")
            case [False, True, False, False, False]:
                col2.write("Button 2 clicked")
            case [False, False, True, False, False]:
                col2.write("Button 3 clicked")
            case [False, False, False, True, False]:
                col2.write("Button 4 clicked")
            case [False, False, False, False, True]:
                col2.write("Button 5 clicked")
