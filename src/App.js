import React, { useRef, useEffect, useState } from 'react';
import * as faceapi from 'face-api.js';
import * as tf from '@tensorflow/tfjs';

function GlowMatchApp() {
  const videoRef = useRef(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [auraMatch, setAuraMatch] = useState(null);
  const [savedLooks, setSavedLooks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const isMounted = useRef(true);

  const startVideo = (videoRef) => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => {
        console.error('Camera access denied:', err);
      });
  };

  const getAuraName = (expression) => {
    const options = [
      { name: "Celestial Queen", score: 94, mood: "neutral", style: "Classic Elegance" },
      { name: "Golden Muse", score: 88, mood: "happy", style: "Radiant Glow" },
      { name: "Electric Vibe", score: 81, mood: "surprised", style: "Bold & Dynamic" },
      { name: "Mystic Flame", score: 76, mood: "angry", style: "Dark & Mysterious" },
      { name: "Moonlight Whisper", score: 85, mood: "sad", style: "Soft & Ethereal" },
      { name: "Enchanted Forest", score: 79, mood: "disgusted", style: "Natural Tones" }
    ];
    const match = options.find(o => o.mood === expression) || options[0];
    return match;
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setUploadedImage(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const saveLook = () => {
    if (auraMatch) {
      setSavedLooks(prev => [...prev, auraMatch]);
    }
  };

  const loadModels = async () => {
    try {
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models'),
        tf.ready()
      ]);
      if (isMounted.current) setIsLoading(false);
    } catch (err) {
      console.error("Failed to load models", err);
      if (isMounted.current) setError("Failed to load models");
    }
  };

  const detectExpression = async (img) => {
    if (!img) return;
    const detections = await faceapi.detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceExpressions();

    if (detections && detections.expressions) {
      const sorted = Object.entries(detections.expressions)
        .sort((a, b) => b[1] - a[1]);
      const topExpression = sorted[0][0];
      const match = getAuraName(topExpression);
      setAuraMatch(match);
    }
  };

  useEffect(() => {
    loadModels();
    startVideo(videoRef);
    return () => {
      isMounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (uploadedImage) {
      const img = new Image();
      img.src = uploadedImage;
      img.onload = () => detectExpression(img);
    }
  }, [uploadedImage]);

  if (isLoading) return <div>Loading models...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div style={{ textAlign: 'center' }} onTouchMove={() => {}}>
      <h1>GlowMatch</h1>
      <h2>{auraMatch?.name || "Detecting your Aura..."}</h2>
      <p>Style Suggestion: {auraMatch?.style || ""}</p>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <button onClick={saveLook}>Save Look</button>

      {/* Optional: Display camera feed */}
      <div style={{ marginTop: '20px' }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          width="320"
          height="240"
          style={{ display: 'block', margin: '10px auto' }}
        />
      </div>

      {/* Saved Looks */}
      <div>
        {savedLooks.map((look, index) => (
          <div key={index}>
            <p>{look.name} – {look.style}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default GlowMatchApp;
