from io import BytesIO
from PIL import Image
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

def predict(image, method):

    files = {'file': image.getvalue() if method=='Image' else image}

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

    * The model use `UNet` as the generator.
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
        rgb = predict(image, method)

        img = transform(image)
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(img.permute(1, 2, 0))
        ax[0].axis('off')
        ax[0].title.set_text('Before')
        ax[1].imshow(rgb)
        ax[1].axis('off')
        ax[1].title.set_text('After')
        st.pyplot(fig)
    else:
        st.warning("Please upload an image")
        st.stop()

elif method == 'URL':

    url = st.text_input('Please enter image URL: ')

    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:

        click = st.button('Colorize')
    if click:

        if url is not None:
            content = requests.get(url).content
            image = Image.open(BytesIO(content))
            rgb = predict(content, method)

            img = transform(image)
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].imshow(img.permute(1, 2, 0))
            ax[0].axis('off')
            ax[0].title.set_text('Before')
            ax[1].imshow(rgb)
            ax[1].axis('off')
            ax[1].title.set_text('After')
            st.pyplot(fig)

        else:
            st.warning("Please enter a URL")
            st.stop()