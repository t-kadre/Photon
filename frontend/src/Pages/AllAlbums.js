import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const AllAlbums = () => {
  const [newAlbumTitle, setNewAlbumTitle] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [albums, setAlbums] = useState([]);
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
      const response = await axios.get(`${process.env.REACT_APP_BASE_URL}/user/albums`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`, // Ensure authToken is available in your component
        },
      });

      if (response.status === 200) { // Assuming 200 OK status for a successful response
        const albums = response.data.map(album => ({ ids: album._id, name: album.album_name }));
        console.log(setAlbums);
        setAlbums(albums);
        setLoading(false);
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
    navigate(`/albums/${albumId}`);
  };

  const handleCreateAlbum = async (e) => {
    e.preventDefault();

    if (newAlbumTitle && newPassword) {
      const newAlbum = {
        album_name: newAlbumTitle,
        password: newPassword,
      };

      try {

        const response = await axios.post(`${process.env.REACT_APP_BASE_URL}/albums`, newAlbum, {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`,
          },
        });

        if (response.status === 201) {
          setNewAlbumTitle('');
        } else {
          console.error('Unexpected response:', response.statusText);
        }
      } catch (error) {
        console.error('Error creating album:', error);
      }
    }
    fetchAllAlbums();
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
        <input
          type="password"
          placeholder="Enter password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
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

          {albums.map((album) => (
            <div
              key={album.ids}
              style={styles.albumCard}
              onClick={() => handleAlbumClick(album.ids)}
            >
              <p>{album.name}</p>
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