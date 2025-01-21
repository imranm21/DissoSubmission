import os
from google.cloud import vision

# Path to your service account key
key_path = "/Users/imranmooraj/Downloads/elated-pathway-435403-s2-3be3d2d2dcd9.json"

# Function to detect faces and emotions using Google Cloud Vision
def detect_faces(image_path):
    # Initialize the Vision client using the service account key directly
    client = vision.ImageAnnotatorClient.from_service_account_json(key_path)

    # Load the image from the file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Call the Vision API to detect faces in the image
    response = client.face_detection(image=image)
    faces = response.face_annotations

    if not faces:
        print("No faces detected.")
        return

    for i, face in enumerate(faces):
        print(f'Face {i + 1}:')

        # Extract likelihood of emotions
        likelihoods = {
            'joy': face.joy_likelihood,
            'sorrow': face.sorrow_likelihood,
            'anger': face.anger_likelihood,
            'surprise': face.surprise_likelihood,
        }

        # Print the likelihood of each emotion
        for emotion, likelihood in likelihoods.items():
            print(f"  {emotion.capitalize()} likelihood: {likelihood}")

    # Handle any errors from the API
    if response.error.message:
        raise Exception(f'{response.error.message}')

# Main function to run the script with a known valid image
def main():
    # Replace this path with the path to your image
    image_path = "/Users/imranmooraj/Downloads/smilingperson.jpeg"
    
    if os.path.exists(image_path):
        print(f"Analyzing face and emotion detection from image: {image_path}")
        detect_faces(image_path)
    else:
        print(f"Image file not found: {image_path}")

if __name__ == "__main__":
    main()
