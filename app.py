%%writefile app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance
import torch
import cv2  # OpenCV for super-resolution and denoising
import numpy as np  # Import numpy to handle array conversions

@st.cache_resource
def load_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        pipe.to("cpu")
        st.warning("CUDA is not available. Running on CPU, which might be slower.")

    return pipe

pipeline = load_pipeline()

def generate_image(prompt):
    if torch.cuda.is_available():
        with torch.autocast("cuda"):
            image = pipeline(prompt).images[0]
    else:
        image = pipeline(prompt).images[0]

    # Convert to OpenCV format for further processing
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Step 1: Upscale using OpenCV's super-resolution
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x4.pb")  # Assuming you have the EDSR model for super-resolution
    sr.setModel("edsr", 4)  # Using 4x upscaling
    upscaled_image_cv = sr.upsample(image_cv)

    # Step 2: Convert back to PIL and apply enhancements
    upscaled_image = Image.fromarray(cv2.cvtColor(upscaled_image_cv, cv2.COLOR_BGR2RGB))
    sharpness_enhancer = ImageEnhance.Sharpness(upscaled_image)
    enhanced_image = sharpness_enhancer.enhance(2.0)  # Increase sharpness

    contrast_enhancer = ImageEnhance.Contrast(enhanced_image)
    final_image = contrast_enhancer.enhance(1.2)  # Slight contrast boost

    return final_image

def main():
    st.title("Arvind Kejriwal Vision Graphic Generator")

    default_prompt = (
        "A realistic, highly detailed photo of Arvind Kejriwal in a Delhi government school classroom. "
        "He is interacting with students of diverse genders and ethnicities, with school elements like desks, "
        "a chalkboard, and educational posters in the background. Kejriwal is wearing his usual attire and has "
        "a friendly, engaging expression. The tone is positive and captures a realistic, lively school environment."
    )

    prompt = st.text_area("Enter your prompt", default_prompt)

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = generate_image(prompt)
            st.image(image, caption="Generated Image", use_column_width=True)
            image.save("Realistic_Kejriwal_vision_graphic.png")
            st.success("Image generated and saved successfully!")
            with open("Realistic_Kejriwal_vision_graphic.png", "rb") as img_file:
                st.download_button("Download Image", data=img_file, file_name="Realistic_Kejriwal_vision_graphic.png")

    st.subheader("Description")
    st.write("This image captures a realistic depiction of Arvind Kejriwal interacting with students, emphasizing his commitment to education in a lively school environment.")

if _name_ == "_main_":
    main()





from pyngrok import ngrok

# Set up ngrok with your authentication token
ngrok.set_auth_token('2o3bjnP2sTbVLKE56SPvPNSwTE1_3i1B75ra2pbjsgcerSsfj')  # Replace with your actual ngrok authtoken

# Create a tunnel to the Streamlit app
public_url = ngrok.connect(8501)
print("Access your app at:", public_url)

# Run the Streamlit app
!streamlit run app.py --server.port 8501 &
