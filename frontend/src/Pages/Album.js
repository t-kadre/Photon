import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useParams, useLocation } from 'react-router-dom';
import Navbar from '../Components/Navbar';
import Modal from '../Components/Modal';
import CustomWebcam from '../Components/CustomWebcam';
import './Album.css';

const Album = () => {
  const location = useLocation();
  const { album_name } = location.state || {};
  const [images, setImages] = useState([]);
  const [cloudinaryImages, setCloudinaryImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false); // State to control modal

  const { albumId } = useParams();
  const authToken = localStorage.getItem('token');

  const cloudName = process.env.REACT_APP_CLOUD_NAME;
  const apiKey = process.env.REACT_APP_CLOUDINARY_API_KEY;
  const apiSecret = process.env.REACT_APP_CLOUDINARY_API_SECRET;

  // useEffect(() => {
  //   fetchAllCloudinaryImages();
  // }, []);

  const handleFileChange = (e) => {
    setImages(e.target.files);
  };

  const handleImageClick = (url) => {
    setSelectedImage(url);
  };

  const closeModal = () => {
    setSelectedImage(null);
  };

  const handleUpload = async () => {
    const urls = [];
    const formData = new FormData();

    for (let i = 0; i < images.length; i++) {
      formData.append('file', images[i]);
      formData.append('upload_preset', 'ai-image-sorter');

      try {
        const response = await axios.post(
          `https://api.cloudinary.com/v1_1/${cloudName}/image/upload`,
          formData
        );
        urls.push(response.data.secure_url);
      } catch (error) {
        console.error('Error uploading image:', error);
      }
    }

    try {
      const backendResponse = await axios.post(`${process.env.REACT_APP_BASE_URL}/albums/${albumId}/upload`,urls,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`,
          },
        }
      );

      console.log('Images and Album details uploaded to backend:', backendResponse.data);
    } catch (backendError) {
      console.error('Error sending data to backend:', backendError);
    }

    fetchAllCloudinaryImages();
  };

  const fetchAllCloudinaryImages = async () => {
    try {
      const response = await axios.get(`${process.env.REACT_APP_BASE_URL}/albums/${albumId}`, {
        params: {
          album_id: albumId,
        },
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`, // Ensure authToken is available in your component
        },
      });

      const existingImageURLs = response.data;
      console.log('Existing images:', existingImageURLs);
      setCloudinaryImages(existingImageURLs);
    } catch (error) {
      console.error('Error fetching images from backend:', error);
    }
  };

  const handleScrollToGallery = () => {
    document.getElementById('gallery-section').scrollIntoView({ behavior: 'smooth' });
    fetchAllCloudinaryImages();
  };

  const handleViewMine = () => {
    setIsModalOpen(true); // Open the modal when "View Mine" is clicked
  };

  const handleImagesReceived = (cloudinaryLinks) => {
    setCloudinaryImages(cloudinaryLinks); // Update cloudinary images with the new links
    setIsModalOpen(false); // Close the modal after images are received
  };

  return (
    <div className="photo-view-page">
      <Navbar />
      <div className="cover-image-container">
        <div className="cover-content">
          <div className="album-name">Memories of {album_name || 'Album'}</div>
          <div className="button-group">
            <div className="view-buttons">
              <button className="action-button" onClick={handleViewMine}>View Mine</button>
              <button className="action-button" onClick={handleScrollToGallery}>View All</button>
            </div>
            <div className="seperation">|</div>
            <div className="upload">
              <input type="file" multiple onChange={handleFileChange} />
              <button className="action-button" onClick={handleUpload}>Upload</button>
            </div>
          </div>
        </div>
      </div>

      <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)}>
        <CustomWebcam
          albumId={albumId}
          authToken={authToken}
          onImagesReceived={handleImagesReceived}
        />
      </Modal>

      <div id="gallery-section" className="gallery-section">

        {/* {cloudinaryImages.length > 0 && (
          <>
            <div className="image-gallery">
              {cloudinaryImages.map((url, index) => (
                <img key={index} src={url} alt={`Uploaded ${index}`} className="gallery-image" />
              ))}
            </div>
          </>
        )} */}
        <div className="image-gallery">
          {cloudinaryImages && (
            <>
              {cloudinaryImages.map((url, index) => (
            <div className="image-container" key={index}>
              <img
                src={url}
                alt={`Uploaded ${index}`}
                className="gallery-image"
                onClick={() => handleImageClick(url)}
              />
              <div className="overlay">
                <button onClick={() => window.open(url, '_blank')}>Download</button>
              </div>
            </div>
          ))}
            </>
          )}
        </div>
        {selectedImage && (
        <div className="modal" onClick={closeModal}>
          <span className="close">&times;</span>
          <img className="modal-content" src={selectedImage} alt="Enlarged view" />
        </div>
      )}
      </div>

      <footer className="footer">
        Powered by Photon
      </footer>
    </div>
  );
};

export default Album;