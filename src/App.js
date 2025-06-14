import React, { useRef, useEffect, useState } from 'react';
import * as faceapi from 'face-api.js';
import * as tf from '@tensorflow/tfjs';

function getAuraName(expression) {
  const options = [
    { name: "Celestial Queen", score: 94, mood: "neutral", style: "Classic Elegance" },
    { name: "Golden Muse", score: 88, mood: "happy", style: "Radiant Glow" },
    { name: "Electric Vibe", score: 81, mood: "surprised", style: "Bold & Dynamic" },
    { name: "Mystic Flame", score: 76, mood: "serious", style: "Dark & Mysterious" }
  ];
  return options.find(opt => opt.mood === expression) || options[0];
}

// Camera Setup
const startVideo = (videoRef) => {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      videoRef.current.srcObject = stream;
      console.log("📸 Camera started");
    })
    .catch((err) => {
      console.error("🚫 Camera error:", err);
    });
};

// Selfie Upload Handling
const handleImageUpload = (event, setUploadedImage) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const image = new Image();
      image.src = e.target.result;

      image.onload = () => {
        setUploadedImage(e.target.result);
        console.log("🖼️ Image fully loaded");
      };
    };
    reader.readAsDataURL(file);
  }
};

// Process Lashes
async function processLashes(imageTensor, lashLength, skinToneFactor) {
  imageTensor = await tf.image.gaussianBlur(imageTensor, [3, 3], 1);
  const gaborFilter = tf.tensor2d([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]);
  const edges = tf.conv2d(imageTensor.mean(2).expandDims(-1), gaborFilter, 1, 'same');
  return edges.mul(tf.scalar(lashLength / 10 * skinToneFactor));
}

// Process Brows
async function processBrows(imageTensor, browThickness, skinToneFactor) {
  imageTensor = await tf.image.gaussianBlur(imageTensor, [3, 3], 1);
  const gaborFilter = tf.tensor2d([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]);
  const edges = tf.conv2d(imageTensor.mean(2).expandDims(-1), gaborFilter, 1, 'same');
  return edges.mul(tf.scalar(browThickness / 10 * skinToneFactor));
}

// Determine Skin Tone Factor
async function getSkinToneFactor(imageTensor) {
  const avgColor = imageTensor.mean().dataSync()[0];
  return 1 + (avgColor - 128) / 256; // Adjust intensity based on skin tone
}

// Render Lashes
function drawLashes(ctx, lashesEdges, curlIntensity) {
  const pixels = lashesEdges.dataSync();
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] > 150) {
      const x = (i % 128) * 5;
      const y = Math.floor(i / 128) * 5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x, y - Math.random() * 10 * curlIntensity);
      ctx.stroke();
    }
  }
}

// Render Brows
function drawBrows(ctx, browsEdges) {
  const pixels = browsEdges.dataSync();
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] > 120) {
      const x = (i % 128) * 5;
      const y = Math.floor(i / 128) * 5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.random() * 5, y - Math.random() * 4);
      ctx.stroke();
    }
  }
}

// Main App
function GlowMatchApp() { console.log("🌟 GlowMatch started");
  const videoRef = useRef();
  const canvasRef = useRef();
  const [auraMatch, setAuraMatch] = useState(null);
  const [lashLength, setLashLength] = useState(10);
  const [browThickness, setBrowThickness] = useState(10);
  const [curlIntensity, setCurlIntensity] = useState(1);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [savedLooks, setSavedLooks] = useState([]);

  const handleTouchMove = (event) => {
    const touch = event.touches[0];
    const changeX = touch.clientX / window.innerWidth;
    const changeY = touch.clientY / window.innerHeight;
    
    setLashLength(Math.max(5, Math.min(20, changeX * 20)));
    setBrowThickness(Math.max(5, Math.min(20, changeY * 20)));
    setCurlIntensity(Math.max(0.5, Math.min(2, (changeY + changeX) / 2)));
  };

  useEffect(() => {
    useEffect(() => {
  Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models').then(() => console.log("✅ Model 1 (ssd_mobilenetv1) loaded")),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models').then(() => console.log("✅ Model 2 (landmark68) loaded")),
    faceapi.nets.faceExpressionNet.loadFromUri('/models').then(() => console.log("✅ Model 3 (expression) loaded")),
    tf.ready().then(() => console.log("✅ TensorFlow ready"))
  ]).then(() => {
    console.log("📸 Starting camera");
    startVideo(videoRef);
  });
}, []);

  useEffect(() => {
    let animationFrame;
    const detectFace = async () => {
      const imageSource = uploadedImage ? new Image() : videoRef.current;
      if (uploadedImage) imageSource.src = uploadedImage;
      
      const detection = await faceapi.detectSingleFace(imageSource, new faceapi.SsdMobilenetv1Options())
        .withFaceLandmarks()
        .withFaceExpressions();

      if (detection) {
        setAuraMatch(getAuraName(Object.keys(detection.expressions).reduce((a, b) =>
          detection.expressions[a] > detection.expressions[b] ? a : b
        )));

        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        const imageTensor = tf.browser.fromPixels(imageSource).resizeBilinear([128, 128]);
        const skinToneFactor = await getSkinToneFactor(imageTensor);

        const lashesEdges = await processLashes(imageTensor, lashLength, skinToneFactor);
        drawLashes(ctx, lashesEdges, curlIntensity);

        const browsEdges = await processBrows(imageTensor, browThickness, skinToneFactor);
        drawBrows(ctx, browsEdges);
      }
      animationFrame = requestAnimationFrame(detectFace);
    };
    detectFace();
    return () => cancelAnimationFrame(animationFrame);
  }, [lashLength, browThickness, curlIntensity, uploadedImage]);

  const saveLook = () => {
    setSavedLooks([...savedLooks, canvasRef.current.toDataURL()]);
  };

  return (
  <div style={{ textAlign: 'center' }} onTouchMove={handleTouchMove}>
    <h1>GlowMatch</h1>
    <h2>{auraMatch?.name || "Detecting your Aura..."}</h2>
    <p>Style Suggestion: {auraMatch?.style}</p>

    <input type="file" accept="image/*" onChange={(e) => handleImageUpload(e, setUploadedImage)} />

    {uploadedImage ? (
      <img src={uploadedImage} alt="Uploaded" width="640" height="480" />
    ) : (
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        width="640"
        height="480"
        style={{ marginTop: '1rem' }}
      />
    )}

    <canvas
      ref={canvasRef}
      width="640"
      height="480"
      style={{ position: 'absolute', top: 0, left: 0 }}
    />

    <button onClick={saveLook} style={{ marginTop: '1rem' }}>
      Save Look
    </button>

    <div style={{ marginTop: '1rem' }}>
      {savedLooks.map((look, index) => (
        <img key={index} src={look} alt={`Look ${index}`} width="100" />
      ))}
    </div>
  </div>
);
  );
}

export default GlowMatchApp;
