import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const AllAlbums = () => {
    const [albums, setAlbums] = useState([]);
    const [newAlbumTitle, setNewAlbumTitle] = useState('');
    const [albumIds, setAlbumIds] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);


    const authToken = localStorage.getItem('token');
    console.log(authToken);
    
    const navigate = useNavigate();

    useEffect(() => {
        fetchAllAlbums();
      }, []);

      const fetchAllAlbums = async () => {
        try {
          const response = await axios.get('https://album-sorter-backend.vercel.app/albums', {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${authToken}`, // Ensure authToken is available in your component
            },
          });
    
          if (response.status === 200) { // Assuming 200 OK status for a successful response
            const ids = response.data; // Assuming the response contains only the album IDs
            setAlbumIds(ids); // Store the album IDs in the state
            setLoading(false);
            console.log('Album IDs:', ids);
          } else {
            console.error('Unexpected response:', response.statusText);
            setLoading(false);
            //setError('An error occurred while fetching albums');
          }
        } catch (error) {
          console.error('Error fetching albums:', error);
          setLoading(false);
          //setError('An error occurred while fetching albums');
        }
      };


    const handleAlbumClick = (albumId) => {
        navigate(`/album/${albumId}`);
    };

    const handleCreateAlbum = async (e) => {
        e.preventDefault();

    if (newAlbumTitle) {
        const newAlbum = {
        userid: 5,
        title: newAlbumTitle,
        };

        try {
        const response = await axios.post('https://album-sorter-backend.vercel.app/albums', newAlbum,{
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`,
            },
        });

        if (response.status === 201) { 
            const createdAlbum = response.data;
            setAlbums([...albums, createdAlbum]);
            setNewAlbumTitle('');
        } else {
            console.error('Unexpected response:', response.statusText);
        }
        } catch (error) {
        console.error('Error creating album:', error);
        }
    }
};



return (
    <div style={styles.albumsContainer}>
      <h2>My Albums</h2>
      <form onSubmit={handleCreateAlbum} style={styles.form}>
        <input
          type="text"
          placeholder="Album Title"
          value={newAlbumTitle}
          onChange={(e) => setNewAlbumTitle(e.target.value)}
          required
          style={styles.input}
        />
        <button type="submit" style={styles.button}>Create Album</button>
      </form>
      {/* Handle loading and error states */}
      {loading ? (
        <div>Loading...</div>
      ) : error ? (
        alert(error) // Use alert for errors
      ) : (
        <div style={styles.albumGrid}>
          {albumIds.map((albumId) => (
            <div
              key={albumId}
              style={styles.albumCard}
              onClick={() => handleAlbumClick(albumId)}
            >
              <p>{albumId}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const styles = {
  albumsContainer: {
    padding: '20px',
  },
  form: {
    marginBottom: '20px',
  },
  input: {
    marginRight: '10px',
    padding: '8px',
    fontSize: '16px',
  },
  button: {
    padding: '8px 16px',
    fontSize: '16px',
    cursor: 'pointer',
  },
  albumGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
    gap: '10px',
  },
  albumCard: {
    textAlign: 'center',
    cursor: 'pointer',
  },
  albumCover: {
    width: '100%',
    height: 'auto',
  },
};

export default AllAlbums;