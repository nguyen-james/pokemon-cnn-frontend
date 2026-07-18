import { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { LABELS, UNKNOWN_SPRITE } from './labels'
import './App.css'

const MODEL_URL = '/model/my-model/model.json'
const INPUT_SIZE = 224
// Fraction of the (square) camera frame captured by the guide box.
const GUIDE_FRAC = 0.5
// Horizontal offset of the guide box, as a fraction of the frame, toward the
// right of the screen. Keeps subjects away from background clutter.
const GUIDE_OFFSET_X = 0.12
// How often we ask the model for a prediction.
const PREDICT_INTERVAL_MS = 350

const spriteUrl = (name) => `/pokemon/${name}.png`
const prettyName = (name) =>
  name
    .split('-')
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(' ')

function App() {
  const videoRef = useRef(null)
  const captureCanvasRef = useRef(null)
  const modelRef = useRef(null)
  const rafRef = useRef(null)
  const lastPredictRef = useRef(0)

  const [status, setStatus] = useState('loading') // loading | ready | running | error
  const [statusMsg, setStatusMsg] = useState('Loading model…')
  const [threshold, setThreshold] = useState(0.6)
  const [prediction, setPrediction] = useState(null) // { label, confidence }

  // Load the TensorFlow.js graph model once.
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        await tf.ready()
        const model = await tf.loadGraphModel(MODEL_URL)
        if (cancelled) return
        modelRef.current = model
        // Warm up so the first real prediction isn't slow.
        tf.tidy(() => {
          const warm = model.predict(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]))
          if (Array.isArray(warm)) warm.forEach((t) => t.dataSync())
          else warm.dataSync()
        })
        setStatus('ready')
        setStatusMsg('Model ready. Start the camera to play!')
      } catch (err) {
        console.error(err)
        setStatus('error')
        setStatusMsg('Failed to load the model. Check the console for details.')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  // Draw the guide square of the current video frame into a 224x224 canvas.
  // Uses cover-crop math so the captured region matches the on-screen box.
  // The video is displayed mirrored (scaleX(-1)), so a rightward on-screen
  // offset maps to a leftward offset in the raw source frame.
  const captureGuide = useCallback(() => {
    const video = videoRef.current
    const canvas = captureCanvasRef.current
    if (!video || !canvas || video.readyState < 2) return null

    const vw = video.videoWidth
    const vh = video.videoHeight
    if (!vw || !vh) return null

    const m = Math.min(vw, vh)
    const side = m * GUIDE_FRAC
    // Mirror flips X: on-screen right offset -> subtract in source space.
    let sx = vw / 2 - GUIDE_OFFSET_X * m - side / 2
    let sy = (vh - side) / 2
    sx = Math.max(0, Math.min(sx, vw - side))
    sy = Math.max(0, Math.min(sy, vh - side))

    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, sx, sy, side, side, 0, 0, INPUT_SIZE, INPUT_SIZE)
    return canvas
  }, [])

  const runPrediction = useCallback(() => {
    const model = modelRef.current
    const canvas = captureGuide()
    if (!model || !canvas) return

    const { label, confidence } = tf.tidy(() => {
      // The model's baked-in Rescaling layer is (x * 2 - 1), which expects
      // [0,1] input and maps it to [-1,1]. So normalize pixels to [0,1] here.
      const input = tf.browser
        .fromPixels(canvas)
        .toFloat()
        .div(255)
        .expandDims(0)
      const out = model.predict(input)
      const probs = Array.isArray(out) ? out[0] : out
      const data = probs.dataSync()
      let bestIdx = 0
      for (let i = 1; i < data.length; i++) {
        if (data[i] > data[bestIdx]) bestIdx = i
      }
      return { label: LABELS[bestIdx], confidence: data[bestIdx] }
    })

    setPrediction({ label, confidence })
  }, [captureGuide])

  // Main loop: throttled predictions while the camera is running.
  const loop = useCallback(() => {
    const now = performance.now()
    if (now - lastPredictRef.current >= PREDICT_INTERVAL_MS) {
      lastPredictRef.current = now
      runPrediction()
    }
    rafRef.current = requestAnimationFrame(loop)
  }, [runPrediction])

  const startCamera = useCallback(async () => {
    try {
      setStatusMsg('Requesting camera…')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 640, height: 480 },
        audio: false,
      })
      const video = videoRef.current
      video.srcObject = stream
      await video.play()
      setStatus('running')
      setStatusMsg('Point the box at a Pokémon!')
      lastPredictRef.current = 0
      rafRef.current = requestAnimationFrame(loop)
    } catch (err) {
      console.error(err)
      setStatus('error')
      setStatusMsg('Could not access the camera. Please grant permission.')
    }
  }, [loop])

  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    const video = videoRef.current
    const stream = video?.srcObject
    if (stream) stream.getTracks().forEach((t) => t.stop())
    if (video) video.srcObject = null
    setStatus('ready')
    setStatusMsg('Camera stopped.')
    setPrediction(null)
  }, [])

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      const stream = videoRef.current?.srcObject
      if (stream) stream.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const isConfident = prediction && prediction.confidence >= threshold
  const shownSprite = isConfident ? prediction.label : UNKNOWN_SPRITE
  const shownName = isConfident ? prettyName(prediction.label) : 'Unknown'
  // Reveal the sprite in full color once we have a prediction (confident
  // Pokémon or the Unown "unknown" fallback); stay a silhouette while idle.
  const revealed = Boolean(prediction)
  const running = status === 'running'

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Who&apos;s That Pokémon?</h1>
        <p className="app__subtitle">
          Powered by a MobileNetV2 CNN running live in your browser with
          TensorFlow.js
        </p>
      </header>

      <main className="stage">
        <section className="camera-card">
          <div className="camera-frame">
            <video
              ref={videoRef}
              className="camera-frame__video"
              playsInline
              muted
            />
            {!running && (
              <div className="camera-frame__placeholder">
                {status === 'loading'
                  ? 'Loading model…'
                  : 'Camera is off'}
              </div>
            )}
            <div
              className={`guide ${running ? 'guide--active' : ''}`}
              style={{
                '--guide-frac': GUIDE_FRAC,
                '--guide-offset-x': GUIDE_OFFSET_X,
              }}
            >
              <span className="guide__corner guide__corner--tl" />
              <span className="guide__corner guide__corner--tr" />
              <span className="guide__corner guide__corner--bl" />
              <span className="guide__corner guide__corner--br" />
            </div>
          </div>

          <div className="controls">
            {!running ? (
              <button
                className="btn btn--primary"
                onClick={startCamera}
                disabled={status === 'loading' || status === 'error'}
              >
                Start Camera
              </button>
            ) : (
              <button className="btn" onClick={stopCamera}>
                Stop Camera
              </button>
            )}

            <label className="threshold">
              <span>
                Confidence threshold: {Math.round(threshold * 100)}%
              </span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </label>
          </div>

          <p className="status">{statusMsg}</p>
        </section>

        <section className="result-card">
          <div className={`silhouette ${revealed ? 'silhouette--revealed' : ''}`}>
            <img
              key={shownSprite}
              src={spriteUrl(shownSprite)}
              alt={shownName}
              className="silhouette__img"
            />
          </div>
          <h2 className="result-card__name">{shownName}</h2>

          {prediction ? (
            <div className="confidence">
              <div className="confidence__bar">
                <div
                  className={`confidence__fill ${
                    isConfident ? 'confidence__fill--ok' : 'confidence__fill--low'
                  }`}
                  style={{ width: `${Math.round(prediction.confidence * 100)}%` }}
                />
              </div>
              <span className="confidence__label">
                {prettyName(prediction.label)} ·{' '}
                {Math.round(prediction.confidence * 100)}%
              </span>
            </div>
          ) : (
            <p className="result-card__hint">
              {running
                ? 'Analyzing…'
                : 'Start the camera to reveal a Pokémon.'}
            </p>
          )}
        </section>
      </main>

      {/* Offscreen canvas used to feed the model. */}
      <canvas
        ref={captureCanvasRef}
        width={INPUT_SIZE}
        height={INPUT_SIZE}
        style={{ display: 'none' }}
      />
    </div>
  )
}

export default App
