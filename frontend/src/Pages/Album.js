import { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';

const Album = () => {
  const [images, setImages] = useState([]);
  const [cloudinaryImages, setCloudinaryImages] = useState([]);
  const { albumId } = useParams();
  
  const cloudName = process.env.REACT_APP_CLOUD_NAME;
  const apiKey = process.env.REACT_APP_CLOUDINARY_API_KEY;
  const apiSecret = process.env.REACT_APP_CLOUDINARY_API_SECRET;

  useEffect(() => {
    fetchAllCloudinaryImages();
  }, []);

  const handleFileChange = (e) => {
    setImages(e.target.files);
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
      const backendResponse = await axios.post(`https://album-sorter-backend.vercel.app/albums/1/upload`, 
        {
        imageUrls: urls,
        },
        {
          headers: {
            Authorization: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMUBleGFtcGxlLmNvbSIsImV4cCI6MTcyNDg0NDI4Mn0.DWNR2Q2KsSCwYk4zUDmdstgncim-Qk1K1Cx63dP69Rw`,
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
      const response = await axios.get(`https://album-sorter-backend.vercel.app/albums/1`, {
        params: {
          album_id: '1',
          password: 'string',
        },
      });
  
      const existingImageURLs = response.data.imageUrls;
      setCloudinaryImages(existingImageURLs);
    } catch (error) {
      console.error('Error fetching images from backend:', error);
    }
  };

  return (
    <div>
      <h2>Upload Multiple Images</h2>
      <input type="file" multiple onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload Images</button>
  
      {cloudinaryImages && (
        <>
          <h3>Uploaded Images</h3>
          <div>
            {cloudinaryImages.map((url, index) => (
              <img key={index} src={url} alt={`Uploaded ${index}`} />
            ))}
            {/* <img src="https://res.cloudinary.com/du1g4j6f8/image/upload/v1724794501/whynegsahth6fveeehk6.jpg" alt="Placeholder 1" />
            <img src="https://res.cloudinary.com/du1g4j6f8/image/upload/v1724794501/whynegsahth6fveeehk6.jpg" alt="Placeholder 2" />
            <img src="https://res.cloudinary.com/du1g4j6f8/image/upload/v1724794501/whynegsahth6fveeehk6.jpg" alt="Placeholder 3" /> */}
          </div>
        </>
      )}
    </div>
  );  
};

export default Album;
