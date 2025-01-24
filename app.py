import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
# title
st.title('My first app')
# simpe text
st.write('Here\'s our first attempt at using data to create a table:')

df = pd.DataFrame({
    "First column": [1, 2, 3, 4],
    "Second column": [10, 20, 30, 40]
})

st.write(df)

#ine chart
line_chart = pd.DataFrame(
    np.random.randn(20, 2),
    columns=['a', 'b']
)

st.line_chart(line_chart)