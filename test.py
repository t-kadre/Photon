import os
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.cluster import DBSCAN
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from PIL import Image, ImageTk


# Initialize the MTCNN face detector and Inception Resnet V1 model
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Function to extract face encodings from an image using MTCNN and Facenet
def extract_face_encodings(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )

    # Detect faces
    boxes, probs = mtcnn.detect(image_rgb)

    if boxes is None:
        print("ello")
        return [],[]

    face_encodings = []
    faces = []
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
            faces.append(face)
            face_encodings.append(face_encoding)

    return face_encodings,faces






def process_images(image_folder):
    encodings = []
    image_paths = []
    facesImg = []
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        face_encodings, faces = extract_face_encodings(image_path)

        # Only add encodings and faces if they exist
        if face_encodings:
            encodings.extend(face_encodings)
            image_paths.extend([image_path] * len(face_encodings))
            facesImg.extend(faces)

    # Ensure that the lengths of encodings and faces match
    assert len(encodings) == len(facesImg), "Mismatch between encodings and faces"

    return np.array(encodings), image_paths, facesImg




# # Function to process all images and extract face encodings
# def process_images(image_folder):
#     encodings = []
#     image_paths = []
#     facesImg = []
#     for image_file in os.listdir(image_folder):
#         image_path = os.path.join(image_folder, image_file)
#         face_encodings,faces = extract_face_encodings(image_path)

#         for encoding in face_encodings:
#             encodings.append(encoding)
#             image_paths.append(image_path)

#         for face in faces:
#             facesImg.append(face)

#     return np.array(encodings), image_paths, np.array(faces)









def cluster_faces(encodings, image_paths, faces, similarity_threshold=0.6):
    # Initialize an empty list to hold clusters
    clusters = []
    
    # Loop through all encodings
    for i, encoding in enumerate(encodings):
        max_similarity = -1
        best_cluster_index = None
        
        # Compare this encoding with existing clusters
        for j, cluster in enumerate(clusters):
            avg_encoding = np.mean(cluster['encodings'], axis=0).reshape(1, -1)
            # Compute the cosine similarity between the encoding and the first encoding in the cluster
            similarity = cosine_similarity([encoding], avg_encoding)[0][0]
            
            # Check if this cluster has the highest similarity so far
            if similarity > max_similarity:
                max_similarity = similarity
                best_cluster_index = j
        
        # If the highest similarity is above the threshold, assign to the best cluster
        if max_similarity > similarity_threshold and best_cluster_index is not None:
            best_cluster = clusters[best_cluster_index]
            best_cluster['encodings'].append(encoding)
            best_cluster['image_paths'].add(os.path.basename(image_paths[i]))
            best_cluster['faces'].append(faces[i])
        else:
            # If not assigned, create a new cluster
            clusters.append({
                'encodings': [encoding],
                'image_paths': {os.path.basename(image_paths[i])},
                'faces': [faces[i]]
            })
    
    # Convert list of clusters to dictionary format
    clusters_dict = {i: cluster for i, cluster in enumerate(clusters)}
    
    return clusters_dict

# # Function to cluster faces and assign images to clusters based on cosine similarity
# def cluster_faces(encodings, image_paths,faces, similarity_threshold=0.5):
#     # Initialize an empty list to hold clusters
#     clusters = []
    
#     # Loop through all encodings
#     for i, encoding in enumerate(encodings):
#         assigned_to_cluster = False
        
#         # Compare this encoding with existing clusters
#         for cluster in clusters:
#             # Compute the cosine similarity between the encoding and the first encoding in the cluster
#             similarity = cosine_similarity([encoding], [cluster['encodings'][0]])[0][0]
            
#             # If similarity is above the threshold, add to this cluster
#             if similarity > similarity_threshold:
#                 cluster['encodings'].append(encoding)
#                 cluster['image_paths'].add(os.path.basename(image_paths[i]))
#                 cluster['faces'].append(faces[i])
#                 assigned_to_cluster = True
#                 break
        
#         # If this encoding wasn't similar to any cluster, create a new cluster
#         if not assigned_to_cluster:
#             clusters.append({
#                 'encodings': [encoding],
#                 'image_paths': {os.path.basename(image_paths[i])},
#                 'faces': [faces[i]]
#             })
    
#     # Convert list of clusters to dictionary format
#     clusters_dict = {i: cluster for i, cluster in enumerate(clusters)}
    
#     return clusters_dict











# # Function to cluster faces and assign images to clusters
# def cluster_faces(encodings, image_paths):
#     # Perform clustering using DBSCAN
#     clustering_model = DBSCAN(eps=0.8, min_samples=1, metric='euclidean')
#     cluster_labels = clustering_model.fit_predict(encodings)

#     # Create a dictionary to hold images and encodings for each cluster
#     clusters = {}
#     for label, encoding, image_path in zip(cluster_labels, encodings, image_paths):
#         if label not in clusters:
#             clusters[label] = {'encodings': [], 'image_paths': set()}  # Use a set to store unique file names
        
#         # Add the image path only if it's not already in the set
#         if os.path.basename(image_path) not in clusters[label]['image_paths']:
#             clusters[label]['encodings'].append(encoding)
#             clusters[label]['image_paths'].add(os.path.basename(image_path))

#     return clusters







def display_cluster_faces(clusters, image_dir='data'):
    final_clusters_image = []

    for cluster_label, data in clusters.items():
        face_list = data['faces']
        if not face_list:
            continue

        # Resize all faces to a fixed size for consistency
        resized_faces = [cv2.resize(face, (100, 100)) for face in face_list if face is not None]

        # Stack faces horizontally until they exceed a certain width, then stack vertically
        if resized_faces:
            max_faces_per_row = 5  # Adjust this value based on the width you want
            rows = []

            for i in range(0, len(resized_faces), max_faces_per_row):
                row_faces = resized_faces[i:i + max_faces_per_row]
                # Pad the row if it has fewer faces than max_faces_per_row
                while len(row_faces) < max_faces_per_row:
                    row_faces.append(np.zeros((100, 100, 3), dtype=np.uint8))
                rows.append(np.hstack(row_faces))

            # Combine all rows into a single image
            cluster_image = np.vstack(rows)

            # Add cluster label at the top of the image
            cluster_image = cv2.copyMakeBorder(cluster_image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.putText(cluster_image, f"Cluster {cluster_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Add this cluster's image to the final_clusters_image list
            final_clusters_image.append(cluster_image)

    if final_clusters_image:
        # Stack all cluster images vertically
        final_image = np.vstack(final_clusters_image)

        # Convert the final image to a format Tkinter can display
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(final_image_rgb)

        # Create the Tkinter window
        root = tk.Tk()
        root.title("Clustered Faces")

        # Create a Canvas widget to support scrolling
        canvas = tk.Canvas(root, width=800, height=600, scrollregion=(0, 0, final_image.shape[1], final_image.shape[0]))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Convert PIL image to ImageTk
        try:
            tk_image = ImageTk.PhotoImage(pil_image)
        except Exception as e:
            print(f"Error converting image to Tkinter format: {e}")
            return

        # Add the image to the Canvas
        canvas.create_image(0, 0, anchor="nw", image=tk_image)

        # Start the Tkinter event loop
        root.mainloop()










# # Function to display cluster number and corresponding faces
# def display_cluster_faces(clusters, image_dir='data'):
#     # base_image = np.zeros((3500, 500, 3), dtype = np.uint8)
#     # display_image = cv2.copyMakeBorder(base_image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
#     x_offset = 10
#     y_offset = 50
#     for cluster_label, data in clusters.items():
#         # Convert the set of image paths to a list to access the first image
#         image_paths_list = list(data['image_paths'])
#         face_list = data['faces']
#         # Create a blank image or use the first image in the cluster as the base
#         if len(image_paths_list) > 0:
#             # base_image_path = os.path.join(image_dir, image_paths_list[0])  # Already a string
#             # base_image = cv2.imread(base_image_path)
#             base_image = np.zeros((350, 500, 3), dtype = np.uint8)
#             if base_image is None:
#                 # print(f"Could not read image: {base_image_path}")
#                 continue

#             display_image = cv2.copyMakeBorder(base_image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
#             # print("empty")
#         else:
#             continue

#         # Add the cluster number as text on the image
#         cv2.putText(display_image, f"Cluster {cluster_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         # Display all faces in this cluster
#         x_offset = 10
#         y_offset =50

#         for face in face_list:
#             # img_path = os.path.join(image_dir, image_path)
#             # print(img_path)
#             img = face
#             if img is None:
#                 # print(f"Could not read image: {img_path}")
#                 continue

#             # face_encodings = extract_face_encodings(img_path)
#             face_img = img  # Use the whole image, or you could crop to just the face area
#             face_resized = cv2.resize(face_img, (100, 100))
#             display_image[y_offset:y_offset+100, x_offset:x_offset+100] = face_resized

#             x_offset += 110  # Move to the next position
#             if x_offset + 100 > display_image.shape[1]:
#                 x_offset = 10
#                 y_offset += 110


#         # Show the final image with the cluster number and all faces
#         cv2.imshow(f"Cluster {cluster_label}", display_image)
#         cv2.waitKey(0)


#     cv2.destroyAllWindows()


# Main execution
def main():
    image_folder = 'data'
    encodings, image_paths, faces = process_images(image_folder)
    clusters = cluster_faces(encodings, image_paths, faces)

    # Save the clusters to a file
    with open('clusters.pkl', 'wb') as f:
        pickle.dump(clusters, f)

    # Print the clusters
    for cluster_label, cluster_data in clusters.items():
        print(f"Cluster {cluster_label}:")
        for image_path in cluster_data['image_paths']:
            print(f"  - {image_path}")
    # Display the cluster faces
    display_cluster_faces(clusters)

if __name__ == "__main__":
    main()
















# import os
# import cv2
# import numpy as np
# from facenet_pytorch import InceptionResnetV1, MTCNN
# from sklearn.cluster import DBSCAN
# import torch
# import pickle  # Import pickle for saving the clusters

# # Initialize the MTCNN face detector and Inception Resnet V1 model
# mtcnn = MTCNN(keep_all=True)
# model = InceptionResnetV1(pretrained='vggface2').eval()

# # Function to extract face encodings from an image using MTCNN and Facenet
# def extract_face_encodings(image_path):
#     image = cv2.imread(image_path)
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

# # Function to process all images and extract face encodings
# def process_images(image_folder):
#     encodings = []
#     image_paths = []

#     for image_file in os.listdir(image_folder):
#         image_path = os.path.join(image_folder, image_file)
#         face_encodings = extract_face_encodings(image_path)

#         for encoding in face_encodings:
#             encodings.append(encoding)
#             image_paths.append(image_path)

#     return np.array(encodings), image_paths

# # Function to cluster faces and assign images to clusters
# def cluster_faces(encodings, image_paths):
#     # Perform clustering using DBSCAN
#     clustering_model = DBSCAN(eps=0.9, min_samples=1, metric='euclidean')
#     cluster_labels = clustering_model.fit_predict(encodings)

#     # Create a dictionary to hold images and encodings for each cluster
#     clusters = {}
#     for label, encoding, image_path in zip(cluster_labels, encodings, image_paths):
#         if label not in clusters:
#             clusters[label] = {'encodings': [], 'image_paths': []}
#         clusters[label]['encodings'].append(encoding)
#         clusters[label]['image_paths'].append(image_path)

#     return clusters

# # Main execution
# def main():
#     image_folder = 'data'
#     encodings, image_paths = process_images(image_folder)
#     clusters = cluster_faces(encodings, image_paths)

#     # Save the clusters to a file
#     with open('clusters.pkl', 'wb') as f:
#         pickle.dump(clusters, f)

#     # Print the clusters
#     for cluster_label, cluster_data in clusters.items():
#         print(f"Cluster {cluster_label}:")
#         for image_path in cluster_data['image_paths']:
#             print(f"  - {image_path}")

# if __name__ == "__main__":
#     main()






















# import os
# import cv2
# import numpy as np
# from facenet_pytorch import InceptionResnetV1, MTCNN
# from sklearn.cluster import DBSCAN
# import torch

# # Initialize the MTCNN face detector and Inception Resnet V1 model
# mtcnn = MTCNN(keep_all=True)
# model = InceptionResnetV1(pretrained='vggface2').eval()

# # Function to extract face encodings from an image using MTCNN and facenet
# def extract_face_encodings(image_path):
#     image = cv2.imread(image_path)
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

# # Function to process all images and extract face encodings
# def process_images(image_folder):
#     encodings = []
#     image_paths = []

#     for image_file in os.listdir(image_folder):
#         image_path = os.path.join(image_folder, image_file)
#         face_encodings = extract_face_encodings(image_path)

#         for encoding in face_encodings:
#             encodings.append(encoding)
#             image_paths.append(image_path)

#     return np.array(encodings), image_paths

# # Function to cluster faces and assign images to clusters
# def cluster_faces(encodings, image_paths):
#     # Perform clustering using DBSCAN
#     clustering_model = DBSCAN(eps=0.6, min_samples=1, metric='euclidean')
#     cluster_labels = clustering_model.fit_predict(encodings)

#     # Create a dictionary to hold images for each cluster
#     clusters = {}
#     for label, image_path in zip(cluster_labels, image_paths):
#         if label not in clusters:
#             clusters[label] = []
#         clusters[label].append(image_path)

#     return clusters

# # Main execution
# def main():
#     image_folder = 'data'
#     encodings, image_paths = process_images(image_folder)
#     clusters = cluster_faces(encodings, image_paths)

#     # Print the clusters
#     for cluster_label, image_list in clusters.items():
#         print(f"Cluster {cluster_label}:")
#         for image_path in image_list:
#             print(f"  - {image_path}")

# if __name__ == "__main__":
#     main()




















# import os
# import cv2
# import numpy as np
# from facenet_pytorch import InceptionResnetV1, MTCNN
# from sklearn.cluster import DBSCAN

# # Initialize the MTCNN face detector and Inception Resnet V1 model
# mtcnn = MTCNN(keep_all=True)
# model = InceptionResnetV1(pretrained='vggface2').eval()

# # Function to extract face encodings from an image using MTCNN and facenet
# def extract_face_encodings(image_path):
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Detect faces
#     boxes, probs = mtcnn.detect(image_rgb)
#     if boxes is None:
#         return []

#     # Extract face encodings
#     face_encodings = []
#     for box in boxes:
#         face = mtcnn.extract(image_rgb, box)
#         face_encoding = model(face.unsqueeze(0)).detach().numpy()
#         face_encodings.append(face_encoding.flatten())

#     return face_encodings

# # Function to process all images and extract face encodings
# def process_images(image_folder):
#     encodings = []
#     image_paths = []

#     for image_file in os.listdir(image_folder):
#         image_path = os.path.join(image_folder, image_file)
#         face_encodings = extract_face_encodings(image_path)

#         for encoding in face_encodings:
#             encodings.append(encoding)
#             image_paths.append(image_path)

#     return np.array(encodings), image_paths

# # Function to cluster faces and assign images to clusters
# def cluster_faces(encodings, image_paths):
#     # Perform clustering using DBSCAN
#     clustering_model = DBSCAN(eps=0.6, min_samples=1, metric='euclidean')
#     cluster_labels = clustering_model.fit_predict(encodings)

#     # Create a dictionary to hold images for each cluster
#     clusters = {}
#     for label, image_path in zip(cluster_labels, image_paths):
#         if label not in clusters:
#             clusters[label] = []
#         clusters[label].append(image_path)

#     return clusters

# # Main execution
# def main():
#     image_folder = 'D:/IIT Guwahati/cpp/New folder/helloworld/WEB_D/AI_Image_Sorter/data'
#     encodings, image_paths = process_images(image_folder)
#     clusters = cluster_faces(encodings, image_paths)

#     # Print the clusters
#     for cluster_label, image_list in clusters.items():
#         print(f"Cluster {cluster_label}:")
#         for image_path in image_list:
#             print(f"  - {image_path}")

# if __name__ == "__main__":
#     main()
