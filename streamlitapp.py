import streamlit as st
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv

# Define a custom sink function to process predictions
def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # Get the text labels for each prediction
    labels = [p["class"] for p in predictions["predictions"]]
    # Load predictions into the Supervision Detections API
    detections = sv.Detections.from_inference(predictions)
    # Annotate the frame using the Supervision annotator
    image = annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    # Display the annotated image
    st.image(image, channels="BGR")

# Initialize the Supervision BoxAnnotator
annotator = sv.BoxAnnotator()

# Define the RTSP URL
rtsp_url2 = st.text_input("Enter RTSP URL", "demo.mov")

# Add the image logo file
logo_image = st.image("Transparent logo.png", width=200)  # Adjust width as needed

# Initialize the InferencePipeline
pipeline = InferencePipeline.init(
    model_id="finalprojectdl/1",
    video_reference=rtsp_url2,
    on_prediction=my_custom_sink,
)

# Start and join the pipeline for processing frames
pipeline.start()

# Create a stop button to stop the pipeline
if st.button("Stop"):
    pipeline.stop()

# Join the pipeline
pipeline.join()
