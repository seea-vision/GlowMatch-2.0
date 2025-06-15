import React, { useRef, useEffect, useState } from 'react';
import * as faceapi from 'face-api.js';
import * as tf from '@tensorflow/tfjs';

function getAuraName(expression) {
  const options = [
    { name: "Celestial Queen", score: 94, mood: "neutral", style: "Classic Elegance" },
    { name: "Golden Muse", score: 88, mood: "happy", style: "Radiant Glow" },
    { name: "Electric Vibe", score: 81, mood: "surprised", style: "Bold & Dynamic" },
    { name: "Mystic Flame", score: 76, mood: "angry", style: "Dark & Mysterious" },
    { name: "Moonlight Whisper", score: 85, mood: "sad", style: "Soft & Ethereal" },
    { name: "Enchanted Forest", score: 79, mood: "disgusted", style: "Natural Tones" },
    { name: "Shadow Dancer", score: 72, mood: "fearful", style: "Edgy Contrast" }
  ];
  return options.find(opt => opt.mood === expression) || options[0];
}

const startVideo = async (videoRef) => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
    return stream;
  } catch (err) {
    console.error("Camera error:", err);
    return null;
  }
};

const handleImageUpload = (event, setUploadedImage) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => setUploadedImage(e.target.result);
    reader.readAsDataURL(file);
  }
};

async function processLashes(imageTensor, lashLength, skinToneFactor) {
  return tf.tidy(() => {
    const blurred = tf.image.gaussianBlur(imageTensor, [3, 3], 1);
    const gaborFilter = tf.tensor2d([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]);
    const edges = tf.conv2d(blurred.mean(2).expandDims(-1), gaborFilter, 1, 'same');
    return edges.mul(tf.scalar(lashLength / 10 * skinToneFactor));
  });
}

async function processBrows(imageTensor, browThickness, skinToneFactor) {
  return tf.tidy(() => {
    const blurred = tf.image.gaussianBlur(imageTensor, [3, 3], 1);
    const gaborFilter = tf.tensor2d([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]);
    const edges = tf.conv2d(blurred.mean(2).expandDims(-1), gaborFilter, 1, 'same');
    return edges.mul(tf.scalar(browThickness / 10 * skinToneFactor));
  });
}

async function getSkinToneFactor(imageTensor) {
  return tf.tidy(() => {
    const avgColor = imageTensor.mean().dataSync()[0];
    return 1 + (avgColor - 128) / 256;
  });
}

function drawLashes(ctx, lashesEdges, curlIntensity) {
  ctx.beginPath();
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 1;
  
  const pixels = lashesEdges.dataSync();
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] > 150) {
      const x = (i % 128) * 5;
      const y = Math.floor(i / 128) * 5;
      ctx.moveTo(x, y);
      ctx.lineTo(x, y - Math.random() * 10 * curlIntensity);
    }
  }
  ctx.stroke();
}

function drawBrows(ctx, browsEdges) {
  ctx.beginPath();
  ctx.strokeStyle = '#333333';
  ctx.lineWidth = 1.5;
  
  const pixels = browsEdges.dataSync();
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] > 120) {
      const x = (i % 128) * 5;
      const y = Math.floor(i / 128) * 5;
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.random() * 5, y - Math.random() * 4);
    }
  }
  ctx.stroke();
}

function GlowMatchApp() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [auraMatch, setAuraMatch] = useState(null);
  const [lashLength, setLashLength] = useState(10);
  const [browThickness, setBrowThickness] = useState(10);
  const [curlIntensity, setCurlIntensity] = useState(1);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [savedLooks, setSavedLooks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const handleTouchMove = (event) => {
    event.preventDefault();
    const touch = event.touches[0];
    const changeX = touch.clientX / window.innerWidth;
    const changeY = touch.clientY / window.innerHeight;
    setLashLength(Math.max(5, Math.min(20, changeX * 20)));
    setBrowThickness(Math.max(5, Math.min(20, changeY * 20)));
    setCurlIntensity(Math.max(0.5, Math.min(2, (changeY + changeX) / 2)));
  };

  useEffect(() => {
    let isMounted = true;
    let stream = null;
    let animationFrameId = null;

    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models'),
          tf.ready()
        ]);
        if (isMounted) setIsLoading(false);
      } catch (err) {
        if (isMounted) setError("Failed to load models");
        console.error("Model loading error:", err);
      }
    };

    const detectFace = async () => {
      if (!isMounted) return;

      try {
        const imageSource = uploadedImage ? new Image() : videoRef.current;
        if (uploadedImage && typeof imageSource.src === 'string') {
          imageSource.src = uploadedImage;
        }

        if ((uploadedImage && !imageSource.complete) || 
            (!uploadedImage && (!videoRef.current || !videoRef.current.readyState))) {
          animationFrameId = requestAnimationFrame(detectFace);
          return;
        }

        const detection = await faceapi
          .detectSingleFace(imageSource)
          .withFaceLandmarks()
          .withFaceExpressions();

        if (detection && isMounted) {
          setAuraMatch(
            getAuraName(
              Object.keys(detection.expressions).reduce((a, b) =>
                detection.expressions[a] > detection.expressions[b] ? a : b
              )
            )
          );

          const ctx = canvasRef.current.getContext('2d');
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          
          const imageTensor = tf.tidy(() => {
            return tf.browser.fromPixels(imageSource).resizeBilinear([128, 128]);
          });
          
          const skinToneFactor = await getSkinToneFactor(imageTensor);
          const lashesEdges = await processLashes(imageTensor, lashLength, skinToneFactor);
          const browsEdges = await processBrows(imageTensor, browThickness, skinToneFactor);
          
          drawLashes(ctx, lashesEdges, curlIntensity);
          drawBrows(ctx, browsEdges);
          
          tf.dispose([imageTensor, lashesEdges, browsEdges]);
        }
      } catch (err) {
        console.error("Detection error:", err);
      }

      animationFrameId = requestAnimationFrame(detectFace);
    };

    loadModels().then(() => {
      if (!uploadedImage) {
        startVideo(videoRef).then((camStream) => {
          stream = camStream;
          if (isMounted) detectFace();
        });
      } else {
        detectFace();
      }
    });

    return () => {
      isMounted = false;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      tf.disposeVariables();
    };
  }, [lashLength, browThickness, curlIntensity, uploadedImage]);

  const saveLook = () => {
    if (canvasRef.current) {
      setSavedLooks(prev => [...prev, canvasRef.current.toDataURL()]);
    }
  };

  if (error) {
    return (
      <div style={{ textAlign: 'center', padding: '20px' }}>
        <h1>Error</h1>
        <p>{error}</p>
        <p>Please check the console for details</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: '20px' }}>
        <h1>Loading...</h1>
        <p>Please wait while we load the models</p>
      </div>
    );
  }

  return (
    <div style={{ textAlign: 'center' }} onTouchMove={handleTouchMove}>
      <h1>GlowMatch</h1>
      <h2>{auraMatch?.name || "Detecting your Aura..."}</h2>
      <p>Style Suggestion: {auraMatch?.style}</p>
      
      <input 
        type="file" 
        accept="image/*" 
        onChange={(e) => handleImageUpload(e, setUploadedImage)}
        style={{ margin: '10px' }}
      />
      
      <div style={{ position: 'relative', margin: '0 auto', width: '640px', height: '480px' }}>
        {!uploadedImage && (
          <video 
            ref={videoRef} 
            autoPlay 
            muted 
            playsInline
            width="640" 
            height="480" 
            style={{ display: 'block' }}
          />
        )}
        <canvas 
          ref={canvasRef} 
          width="640" 
          height="480" 
          style={{ 
            position: 'absolute', 
            top: 0, 
            left: 0,
            pointerEvents: 'none'
          }} 
        />
      </div>
      
      <button 
        onClick={saveLook} 
        style={{ 
          margin: '20px',
          padding: '10px 20px',
          fontSize: '16px',
          cursor: 'pointer'
        }}
      >
        Save Look
      </button>
      
      {savedLooks.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h3>Saved Looks:</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
            {savedLooks.map((look, i) => (
              <img 
                key={i} 
                src={look} 
                alt={`Saved look ${i}`} 
                width="120" 
                height="90" 
                style={{ margin: '5px', border: '1px solid #ddd' }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default GlowMatchApp;
