
import { Image, Video, Transformation, CloudinaryContext } from 'cloudinary-react';
import cloudinary from 'cloudinary-core';
import { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from '../Components/Navbar';
import AllAlbums from '../Components/AllAlbums';

const Home = () => {

  return (
    <div>
        <Navbar />
      <AllAlbums />
    </div>
  );
};

export default Home;



