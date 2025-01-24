import streamlit as st
st.title("Streamlit Widgets")


user_name = st.text_input("Enter your name")

if user_name:
    st.write(f"Hello {user_name}")

# slider
age = st.slider("Enter Your age",0,100,25)
st.write(f"Your age is {age}")
# selectbox
options = ["Python","Java","C++","C#"]
selected_options = st.selectbox("Select your favourite languge",options)
st.write(f"Your favourite language is {selected_options}")

uploaded_file = st.file_uploader("Choose a file",type=["jpg","png","jpeg"])
st.write(uploaded_file)
