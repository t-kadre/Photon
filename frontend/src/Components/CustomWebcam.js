import React, { useRef, useState, useCallback, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const CustomWebcam = ({ albumId, authToken, onImagesReceived }) => {
  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(null);
  const [imgUrl, setImgUrl] = useState(null);
  const cloudName = process.env.REACT_APP_CLOUD_NAME;


  const capture = useCallback(async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);
    const formData = new FormData();
    formData.append("file", imageSrc);
    formData.append("upload_preset", "ai-image-sorter");

    try {
      const response = await axios.post(
        `https://api.cloudinary.com/v1_1/${cloudName}/image/upload`,
        formData
      );
      setImgUrl(response.data.secure_url);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  }, [albumId, authToken, onImagesReceived]);

  useEffect(() => {
    const fetchImages = async () => {
      if (imgUrl) {
        try {
          const response = await axios.get(
            `${process.env.REACT_APP_BASE_URL}/albums/${albumId}/find-my-images`,
            {
              params: {
                album_id: albumId,
                url: imgUrl,
              },
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`,
              },
            }
          );
  
          const cloudinaryLinks = response.data;
          if (onImagesReceived) {
            onImagesReceived(cloudinaryLinks);
          }
        } catch (error) {
          console.error("Error fetching images:", error);
        }
      }
    };
  
    fetchImages();
  }, [imgUrl, onImagesReceived]);
  

  return (
    <div className="container">
      {imgSrc ? (
        <img src={imgSrc} alt="webcam" />
      ) : (
        <Webcam height={600} width={600} ref={webcamRef} />
      )}
      <div className="btn-container">
        <button onClick={capture}>Capture photo</button>
      </div>
    </div>
  );
};

export default CustomWebcam;
