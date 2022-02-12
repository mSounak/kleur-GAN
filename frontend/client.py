from PIL import Image
import streamlit as st
import requests
import numpy as np


st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)


def predict(image):

    files = {'file': image.getvalue()}

    res = requests.post('http://localhost:8000/predict', files=files)

    rest = res.json()

    image = rest['image']
    image = np.array(image)
    return image

# Side Bar
with st.sidebar:
    st.title("Kleur-GAN 🎨")
    st.write("""
    This is a colorization app using the GAN Pix2Pix model.

    * The model use UNet as the generator.
    * The model is trained on `Flicker8k` dataset.

    *So... If you see any inconsistencies, Just know that this is trained only on 6k images.*
    """)

    st.write("")
    st.write("")
    st.write('## Choose Input type')
    method = st.radio(label='',options=['Image', 'URL'])

    st.write("")
    st.write("")
    st.markdown('To know more about this app, visit [GitHub](https://github.com/mSounak/kleur-GAN/)')

# Main body
st.title("Kleur-GAN 🖌️🎨")
st.write("")
st.write("")
st.markdown("### Input your image")
if method == 'Image':
    image = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])
    if image is not None:
        rgb = predict(image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(image), caption='Before', use_column_width=True)
        with col2:
            
            st.image(rgb, caption='After', use_column_width=True)