import os
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.cluster import DBSCAN
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors



# Initialize the MTCNN face detector and Inception Resnet V1 model
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Function to extract face encodings from an image using MTCNN and facenet
def extract_face_encodings(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs = mtcnn.detect(image_rgb)
    if boxes is None:
        return []

    face_encodings = []
    for box in boxes:
        # Ensure the coordinates are within the image bounds
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

        # Check if the bounding box is valid
        if x2 > x1 and y2 > y1:
            face = image_rgb[y1:y2, x1:x2]

            # Resize the face to the required size (160x160) and convert to tensor
            face_resized = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()

            # Normalize the face tensor
            face_tensor = (face_tensor - 127.5) / 128.0

            # Get the face encoding using the Facenet model
            with torch.no_grad():
                face_encoding = model(face_tensor).numpy().flatten()

            face_encodings.append(face_encoding)

    return face_encodings




# Function to find the nearest cluster for an input encoding using KNN
def find_nearest_cluster(input_encoding, clusters):
    encodings = []
    labels = []
    for cluster_label, data in clusters.items():
        encodings.extend(data['encodings'])
        labels.extend([cluster_label] * len(data['encodings']))

    # Convert lists to numpy arrays
    encodings = np.array(encodings)
    labels = np.array(labels)
    n_neighbors=int(np.sqrt(len(set(labels))))
    # Initialize the KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(encodings)

    # Find the nearest neighbors
    distances, indices = knn.kneighbors([input_encoding])

    # Get the labels of the nearest neighbors
    nearest_labels = labels[indices[0]]
    print(labels[indices])

    # Find the most common label (cluster)
    nearest_cluster = np.bincount(nearest_labels).argmax()

    return nearest_cluster























# # Function to find the nearest cluster for an input encoding
# def find_nearest_cluster(input_encoding, clusters):
#     min_distance = float('inf')
#     nearest_cluster = None

#     for cluster_label, encodings in clusters.items():
#         for encoding in encodings['encodings']:
#             distance = np.linalg.norm(input_encoding - encoding)
#             # distance = 1- cosine_similarity([input_encoding], [encoding])[0][0]
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_cluster = cluster_label

#     return nearest_cluster


# Function to search through the wedding photos using query_embedding
def search_wedding_photos(query_embedding, clusters, image_dir='data'):
    nearest_cluster = find_nearest_cluster(query_embedding, clusters)

    if nearest_cluster is not None:
        print(f"Searching for similar faces in cluster {nearest_cluster}:")

        # Convert the query embedding to a 2D array for cosine similarity calculation
        query_embedding = np.array(query_embedding).reshape(1, -1)

        for encoding, image_path in zip(clusters[nearest_cluster]['encodings'], clusters[nearest_cluster]['image_paths']):
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, np.array(encoding).reshape(1, -1))[0][0]

            # Check if the similarity is greater than 60%
            if similarity > 0.5:
                print(f"Similarity: {similarity:.2f} - Displaying image: {image_path}")
                img = cv2.imread(os.path.join(image_dir, os.path.basename(image_path)))
                img = cv2.resize(img,(600,600))
                cv2.imshow('Matching Photo', img)
                cv2.waitKey(0)

    else:
        print("No similar faces found.")

    cv2.destroyAllWindows()




# # Function to search through the wedding photos using query_embedding
# def search_wedding_photos(query_embedding, clusters, image_dir='data'):
#     nearest_cluster = find_nearest_cluster(query_embedding, clusters)

#     if nearest_cluster is not None:
#         print(f"Found similar faces in cluster {nearest_cluster}:")
#         for image_path in clusters[nearest_cluster]['image_paths']:
#             img = cv2.imread(os.path.join(image_dir, os.path.basename(image_path)))
#             cv2.imshow('Matching Photo', img)
#             cv2.waitKey(0)
#     else:
#         print("No similar faces found.")

#     cv2.destroyAllWindows()

# Function to capture a face from the webcam and return its encoding
def capture_face_encoding():
    cap = cv2.VideoCapture(0)
    query_encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Webcam', frame)

        # Wait for 'q' key press to capture the image and process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
            encodings = extract_face_encodings(frame)
            if encodings:
                query_encoding = encodings[0]
            break

    cap.release()
    cv2.destroyAllWindows()
    return query_encoding

# Main execution
def main():
    # Load the clusters from the file
    clusters = {}
    with open('clusters.pkl', 'rb') as f:
        clusters = pickle.load(f)

    # Capture the query face encoding from the webcam
    query_encoding = capture_face_encoding()

    if query_encoding is not None:
        search_wedding_photos(query_encoding, clusters)
    else:
        print("No face detected.")

if __name__ == "__main__":
    main()


















# import os
# import cv2
# import numpy as np
# from facenet_pytorch import InceptionResnetV1, MTCNN
# from sklearn.cluster import DBSCAN
# import torch
# import pickle


# # Initialize the MTCNN face detector and Inception Resnet V1 model
# mtcnn = MTCNN(keep_all=True)
# model = InceptionResnetV1(pretrained='vggface2').eval()

# # Function to extract face encodings from an image using MTCNN and facenet
# def extract_face_encodings(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Detect faces
#     boxes, probs = mtcnn.detect(image_rgb)
#     if boxes is None:
#         return []

#     face_encodings = []
#     for box in boxes:
#         # Ensure the coordinates are within the image bounds
#         x1, y1, x2, y2 = map(int, box)
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

#         # Check if the bounding box is valid
#         if x2 > x1 and y2 > y1:
#             face = image_rgb[y1:y2, x1:x2]

#             # Resize the face to the required size (160x160) and convert to tensor
#             face_resized = cv2.resize(face, (160, 160))
#             face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()

#             # Normalize the face tensor
#             face_tensor = (face_tensor - 127.5) / 128.0

#             # Get the face encoding using the Facenet model
#             with torch.no_grad():
#                 face_encoding = model(face_tensor).numpy().flatten()

#             face_encodings.append(face_encoding)

#     return face_encodings

# # Function to find the nearest cluster for an input encoding
# def find_nearest_cluster(input_encoding, clusters):
#     min_distance = float('inf')
#     nearest_cluster = None

#     for cluster_label, encodings in clusters.items():
#         for encoding in encodings['encodings']:
#             distance = np.linalg.norm(input_encoding - encoding)
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_cluster = cluster_label

#     return nearest_cluster

# # Function to search through the wedding photos using query_embedding
# def search_wedding_photos(query_embedding, clusters, image_dir='data'):
#     nearest_cluster = find_nearest_cluster(query_embedding, clusters)

#     if nearest_cluster is not None:
#         print(f"Found similar faces in cluster {nearest_cluster}:")
#         for image_path in clusters[nearest_cluster]['image_paths']:
#             img = cv2.imread(os.path.join(image_dir, os.path.basename(image_path)))
#             cv2.imshow('Matching Photo', img)
#             cv2.waitKey(0)
#     else:
#         print("No similar faces found.")

#     cv2.destroyAllWindows()

# # Function to capture a face from the webcam and return its encoding
# def capture_face_encoding():
#     cap = cv2.VideoCapture(0)
#     query_encoding = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         cv2.imshow('Webcam', frame)
#         encodings = extract_face_encodings(frame)

#         if encodings:
#             query_encoding = encodings[0]
#             break

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return query_encoding

# # Main execution
# def main():
#     # Load the clusters from the file
#     clusters = {}
#     with open('clusters.pkl', 'rb') as f:
#         clusters = pickle.load(f)

#     # Capture the query face encoding from the webcam
#     query_encoding = capture_face_encoding()

#     if query_encoding is not None:
#         search_wedding_photos(query_encoding, clusters)
#     else:
#         print("No face detected.")

# if __name__ == "__main__":
#     main()