import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    processor, model = load_model()

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    inputs = processor(image, return_tensors="pt")

    with st.spinner("Generating caption..."):
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.success(f"Caption: {caption}")