import { Image, Video, Transformation, CloudinaryContext } from 'cloudinary-react';
import cloudinary from 'cloudinary-core';
import { useState, useEffect } from 'react';
import axios from 'axios';

const Home = () => {
  const [images, setImages] = useState([]);
  const [imageURLs, setImageURLs] = useState([]);
  const [cloudinaryImages, setCloudinaryImages] = useState([]);
  
  const cloudName = 'du1g4j6f8';
  const apiKey = '775349273443457';
  const apiSecret = 'LpwP-jH6DnhyvVieJ5nfbPKp2wI';

  useEffect(() => {
    fetchAllCloudinaryImages();
  }, []);

  const handleFileChange = (e) => {
    setImages(e.target.files);
  };

  const handleUpload = async () => {
    const urls = [...imageURLs];
    const formData = new FormData();
    
    for (let i = 0; i < images.length; i++) {
      formData.append('file', images[i]);
      formData.append('upload_preset', 'ai-image-sorter'); // Replace with your upload preset

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

    setImageURLs(urls);
    fetchAllCloudinaryImages(); // Refresh the list of all images after upload
  };

  const fetchAllCloudinaryImages = async () => {
    try {
      const response = await axios.get(
        `https://api.cloudinary.com/v1_1/${cloudName}/resources/image`,
        {
          auth: {
            username: apiKey,
            password: apiSecret,
          }
        }
      );
      const existingImageURLs = response.data.resources.map(
        (resource) => resource.secure_url
      );
      setCloudinaryImages(existingImageURLs);
    } catch (error) {
      console.error('Error fetching images:', error);
    }
  };

  return (
    <div>
      <h2>Upload Multiple Images</h2>
      <input type="file" multiple onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload Images</button>

      <h3>Uploaded Images</h3>
      <div>
        {cloudinaryImages.concat(imageURLs).map((url, index) => (
          <img key={index} src={url} alt={`Uploaded ${index}`} width="200" />
        ))}
      </div>
    </div>
  );
};

export default Home;
