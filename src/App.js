
import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import * as tf from '@tensorflow/tfjs';

function App() {
  const videoRef = useRef();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);

  const startVideo = () => {
    navigator.mediaDevices.getUserMedia({ video: {} })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error('Camera error:', err);
        setError('Camera access denied or not supported');
      });
  };

  const loadModels = async () => {
    try {
      console.log('Starting model load...');
      setError(null);

      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models'),
        tf.ready()
      ]);

      console.log('✅ All models loaded successfully');
      setIsLoading(false);
      startVideo();
    } catch (err) {
      console.error('❌ Failed to load models:', err);
      setError('Failed to load models: ' + err.message);
      setIsLoading(false);
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log('📸 Selfie selected:', file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  return (
    <div style={{ textAlign: 'center', padding: 30 }}>
      <h1>GlowMatch</h1>
      {isLoading && <p>Loading models...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {!isLoading && (
        <>
          <p>Upload a selfie or use your camera:</p>
          <input type="file" accept="image/*" onChange={handleImageUpload} />
          <br /><br />
          {uploadedImage && <img src={uploadedImage} alt="Selfie" width={200} />}
          <br /><br />
          <video ref={videoRef} autoPlay muted width={300} height={225} style={{ border: '1px solid black' }} />
        </>
      )}
    </div>
  );
}

export default App;
