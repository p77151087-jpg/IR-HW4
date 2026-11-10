import streamlit as st
from ir_hw4.ui import render


def main():
    st.set_page_config(page_title="Reuters Lab · IR HW4", page_icon="📰", layout="wide")
    render()


if __name__ == "__main__":
    main()
