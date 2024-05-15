# Import necessary modules
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv

# Define the RTSP URL you want to use
rtsp_url = "rtsp://clipper-1.zerolatency.tv:554/rcs/feed1satvmix"

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
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

# Initialize the Supervision BoxAnnotator
annotator = sv.BoxAnnotator()

# Initialize the video capture object for HD resolution
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width to HD resolution (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height to HD resolution (1920x1080)

# Initialize the InferencePipeline (this assumes a class like InferencePipeline exists in your codebase)
pipeline = InferencePipeline.init(
    model_id="finalprojectdl/1",
    video_reference=cap,  # Use the modified video capture object
    on_prediction=my_custom_sink,
)

# Start and join the pipeline for processing frames
pipeline.start()
pipeline.join()

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
