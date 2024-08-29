import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useParams, useLocation } from 'react-router-dom';
import Navbar from '../Components/Navbar';
import './Album.css';

const Album = () => {
  const location = useLocation();
  const { album_name } = location.state || {};
  const [images, setImages] = useState([]);
  const [cloudinaryImages, setCloudinaryImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const { albumId } = useParams();
  const authToken = localStorage.getItem('token');

  const cloudName = process.env.REACT_APP_CLOUD_NAME;
  const apiKey = process.env.REACT_APP_CLOUDINARY_API_KEY;
  const apiSecret = process.env.REACT_APP_CLOUDINARY_API_SECRET;

  useEffect(() => {
    fetchAllCloudinaryImages();
  }, []);

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

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setCameraStream(stream);
      videoRef.current.srcObject = stream;
      setCameraEnabled(true);
    } catch (error) {
      console.error('Error accessing the camera:', error);
    }
  };

  const captureImage = () => {
    const context = canvasRef.current.getContext('2d');
    context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    const dataUrl = canvasRef.current.toDataURL('image/png');
    return dataUrl;
  };

  const handleViewMine = async () => {
    if (!cameraEnabled) {
      startCamera();
    } else {
      const capturedImage = captureImage();
      try {
        const response = await axios.get(
          `${process.env.REACT_APP_BASE_URL}/albums/${albumId}/find-my-images`,
          {
            params: {
              album_id: albumId,
              captured_image: capturedImage,
            },
            headers: {
              'Authorization': `Bearer ${authToken}`,
            },
          }
        );

        const myImages = response.data; // Assuming it returns a list of image URLs
        setCloudinaryImages(myImages);
        if (cameraStream) {
          cameraStream.getTracks().forEach(track => track.stop());
        }
        setCameraEnabled(false);
      } catch (error) {
        console.error('Error fetching my images:', error);
      }
    }
  };

  const handleScrollToGallery = () => {
    document.getElementById('gallery-section').scrollIntoView({ behavior: 'smooth' });
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

      {cameraEnabled && (
        <div className="camera-section">
          <video ref={videoRef} autoPlay className="camera-view"></video>
          <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480"></canvas>
        </div>
      )}

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
