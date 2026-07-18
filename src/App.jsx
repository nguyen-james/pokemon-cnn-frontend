import { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { LABELS, UNKNOWN_SPRITE } from './labels'
import './App.css'

const MODEL_URL = '/model/my-model/model.json'
const INPUT_SIZE = 224
const GUIDE_FRAC = 0.5
const GUIDE_OFFSET_X = 0.12
const PREDICT_INTERVAL_MS = 350
const STABILITY_WINDOW = 5
const STABILITY_REQUIRED = 3
const HYSTERESIS = 0.05

const spriteUrl = (name) => `/pokemon/${name}.png`
const prettyName = (name) =>
  name
    .split('-')
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(' ')

function topTwo(probs) {
  let bestIdx = 0
  let secondIdx = 1
  if (probs[1] > probs[0]) {
    bestIdx = 1
    secondIdx = 0
  }
  for (let i = 2; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) {
      secondIdx = bestIdx
      bestIdx = i
    } else if (probs[i] > probs[secondIdx]) {
      secondIdx = i
    }
  }
  return { bestIdx, secondIdx }
}

function stableLabel(recentLabels) {
  const counts = new Map()
  for (const label of recentLabels) {
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  let winner = recentLabels[recentLabels.length - 1]
  let bestCount = 0
  for (const [label, count] of counts) {
    if (count > bestCount) {
      bestCount = count
      winner = label
    }
  }
  return bestCount >= STABILITY_REQUIRED ? winner : null
}

function passesThresholds(confidence, margin, thresholds, wasConfident) {
  const { confidenceMin, marginMin } = thresholds
  if (wasConfident) {
    return (
      confidence >= confidenceMin - HYSTERESIS &&
      margin >= marginMin - HYSTERESIS
    )
  }
  return (
    confidence >= confidenceMin + HYSTERESIS &&
    margin >= marginMin + HYSTERESIS
  )
}

function App() {
  const videoRef = useRef(null)
  const captureCanvasRef = useRef(null)
  const modelRef = useRef(null)
  const rafRef = useRef(null)
  const lastPredictRef = useRef(0)
  const labelHistoryRef = useRef([])
  const wasConfidentRef = useRef(false)
  const displayedRef = useRef(null)
  const confidenceMinRef = useRef(0.6)
  const marginMinRef = useRef(0.15)

  const [status, setStatus] = useState('loading')
  const [statusMsg, setStatusMsg] = useState('Loading model…')
  const [confidenceMin, setConfidenceMin] = useState(0.6)
  const [marginMin, setMarginMin] = useState(0.15)
  const [prediction, setPrediction] = useState(null)
  const [isConfident, setIsConfident] = useState(false)

  confidenceMinRef.current = confidenceMin
  marginMinRef.current = marginMin

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        await tf.ready()
        const model = await tf.loadGraphModel(MODEL_URL)
        if (cancelled) return
        modelRef.current = model
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

  const captureGuide = useCallback(() => {
    const video = videoRef.current
    const canvas = captureCanvasRef.current
    if (!video || !canvas || video.readyState < 2) return null

    const vw = video.videoWidth
    const vh = video.videoHeight
    if (!vw || !vh) return null

    const m = Math.min(vw, vh)
    const side = m * GUIDE_FRAC
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

    const raw = tf.tidy(() => {
      const input = tf.browser
        .fromPixels(canvas)
        .toFloat()
        .div(255)
        .expandDims(0)
      const out = model.predict(input)
      const probs = Array.isArray(out) ? out[0] : out
      const data = probs.dataSync()
      const { bestIdx, secondIdx } = topTwo(data)
      const confidence = data[bestIdx]
      const secondConfidence = data[secondIdx]
      return {
        label: LABELS[bestIdx],
        confidence,
        secondConfidence,
        margin: confidence - secondConfidence,
      }
    })

    labelHistoryRef.current = [
      ...labelHistoryRef.current.slice(-(STABILITY_WINDOW - 1)),
      raw.label,
    ]
    const smoothedLabel =
      stableLabel(labelHistoryRef.current) ??
      displayedRef.current?.label ??
      raw.label

    const confident = passesThresholds(
      raw.confidence,
      raw.margin,
      {
        confidenceMin: confidenceMinRef.current,
        marginMin: marginMinRef.current,
      },
      wasConfidentRef.current,
    )
    wasConfidentRef.current = confident

    const next = {
      label: smoothedLabel,
      confidence: raw.confidence,
      secondConfidence: raw.secondConfidence,
      margin: raw.margin,
    }

    const prev = displayedRef.current
    const changed =
      !prev ||
      prev.label !== next.label ||
      prev.confidence !== next.confidence ||
      prev.margin !== next.margin ||
      prev.confident !== confident

    if (changed) {
      displayedRef.current = { ...next, confident }
      setPrediction(next)
      setIsConfident(confident)
    }
  }, [captureGuide])

  const loop = useCallback(() => {
    const now = performance.now()
    if (now - lastPredictRef.current >= PREDICT_INTERVAL_MS) {
      lastPredictRef.current = now
      runPrediction()
    }
    rafRef.current = requestAnimationFrame(loop)
  }, [runPrediction])

  const resetPredictionState = useCallback(() => {
    labelHistoryRef.current = []
    wasConfidentRef.current = false
    displayedRef.current = null
    setPrediction(null)
    setIsConfident(false)
  }, [])

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
      resetPredictionState()
      setStatus('running')
      setStatusMsg('Fill the yellow box with one Pokémon!')
      lastPredictRef.current = 0
      rafRef.current = requestAnimationFrame(loop)
    } catch (err) {
      console.error(err)
      setStatus('error')
      setStatusMsg('Could not access the camera. Please grant permission.')
    }
  }, [loop, resetPredictionState])

  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    const video = videoRef.current
    const stream = video?.srcObject
    if (stream) stream.getTracks().forEach((t) => t.stop())
    if (video) video.srcObject = null
    resetPredictionState()
    setStatus('ready')
    setStatusMsg('Camera stopped.')
  }, [resetPredictionState])

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      const stream = videoRef.current?.srcObject
      if (stream) stream.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const shownSprite = isConfident && prediction ? prediction.label : UNKNOWN_SPRITE
  const shownName = isConfident && prediction
    ? prettyName(prediction.label)
    : prediction
      ? 'Unknown'
      : '—'
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
        <p className="app__instructions">
          Fill the yellow guide box with a single Pokémon toy, card, or screen.
          Keep the background out of the box for the best results.
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
            {running && (
              <div
                className="guide guide--active"
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
            )}
            {running && (
              <div className="model-preview" aria-label="Model input preview">
                <span className="model-preview__label">Model input</span>
                <canvas
                  ref={captureCanvasRef}
                  width={INPUT_SIZE}
                  height={INPUT_SIZE}
                  className="model-preview__canvas"
                />
              </div>
            )}
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
                Min confidence: {Math.round(confidenceMin * 100)}%
              </span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={confidenceMin}
                onChange={(e) => setConfidenceMin(Number(e.target.value))}
              />
            </label>

            <label className="threshold">
              <span>
                Min margin: {Math.round(marginMin * 100)}%
              </span>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.01"
                value={marginMin}
                onChange={(e) => setMarginMin(Number(e.target.value))}
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
          <h2 className="result-card__name" aria-live="polite">
            {shownName}
          </h2>

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
                {Math.round(prediction.confidence * 100)}% top ·{' '}
                {Math.round(prediction.margin * 100)}% margin
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

      {!running && (
        <canvas
          ref={captureCanvasRef}
          width={INPUT_SIZE}
          height={INPUT_SIZE}
          style={{ display: 'none' }}
          aria-hidden="true"
        />
      )}
    </div>
  )
}

export default App
