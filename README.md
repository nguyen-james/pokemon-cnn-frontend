# Who's That Pokémon?

A browser-based **"Who's That Pokémon?"** game that identifies Gen 1 Pokémon in real time using a MobileNetV2 classifier running entirely in the client with [TensorFlow.js](https://www.tensorflow.org/js).

Point your camera at a Pokémon toy, card, or screen image inside the yellow guide box — the app predicts the species and reveals its sprite. If the model is not confident enough, it shows **Unown** instead.

![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=tensorflow&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)

## Features

- **Live webcam inference** — no server required after the page loads
- **151-class Gen 1 classifier** — MobileNetV2 fine-tuned on Pokémon sprites
- **Guide box overlay** — crops a focused region to reduce background interference
- **Confidence gating** — uses both top softmax probability and top-minus-second margin (matching a Python/OpenCV reference pipeline)
- **Prediction smoothing** — temporal label stability and threshold hysteresis to reduce flicker
- **Model input preview** — live 224×224 thumbnail showing exactly what the network sees
- **Adjustable thresholds** — tune min confidence and min margin with on-screen sliders

## Quick start

**Requirements:** Node.js 18+, a modern browser with camera access (Chrome, Edge, Firefox, Safari). Camera APIs require `localhost` or HTTPS.

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173), click **Start Camera**, and fill the yellow box with a single Pokémon.

### Production build

```bash
npm run build
npm run preview
```

The built app is output to `dist/`. Deploy that folder to any static host (Netlify, Vercel, GitHub Pages, etc.).

## How to use

1. Allow camera permission when prompted.
2. Place **one** Pokémon subject inside the yellow guide box — toy, trading card, phone screen, etc.
3. Keep extra background clutter **outside** the box for best results.
4. Watch the result panel: a confident match reveals the Pokémon sprite and name; a low-confidence guess shows the Unown sprite with **Unknown**.
5. Adjust **Min confidence** and **Min margin** if predictions feel too strict or too loose.

The small **Model input** preview in the corner of the camera feed shows the exact 224×224 crop sent to the network.

## How it works

```text
Webcam frame
    → crop guide-box region (mirrored preview, unmirrored capture math)
    → resize to 224×224
    → normalize pixels to [0, 1]
    → MobileNetV2 graph model (TensorFlow.js)
    → softmax over 151 classes
    → smoothing + margin/confidence thresholds
    → sprite + label (or Unown fallback)
```

### Model

| Property | Value |
|----------|-------|
| Architecture | MobileNetV2 (224×224 input) |
| Output | 151 softmax classes (Gen 1, alphabetical label order) |
| Format | TensorFlow.js graph model (`tf.loadGraphModel`) |
| Location | `public/model/my-model/` (`model.json` + 3 weight shards) |
| Preprocessing | Model includes a baked-in `Rescaling(2, -1)` layer — feed **[0, 1]** pixels (browser divides by 255) |

### Confidence logic

A prediction is accepted when **both** conditions pass (with ±5% hysteresis to prevent flicker):

- **Top confidence** ≥ min confidence threshold (default 60%)
- **Margin** (top − second softmax score) ≥ min margin threshold (default 15%)

The displayed label must also appear in at least **3 of the last 5** inference frames before updating.

## Project structure

```text
pokemon-cnn-frontend/
├── public/
│   ├── model/my-model/     # TensorFlow.js model (model.json + shards)
│   └── pokemon/            # 151 Gen 1 sprites + unown-f.png fallback
├── src/
│   ├── App.jsx             # Camera, inference loop, UI
│   ├── App.css             # Layout and styling
│   ├── labels.js           # 151 class names (alphabetical, matches model)
│   ├── main.jsx
│   └── index.css
├── index.html
├── package.json
└── vite.config.js
```

## Configuration

Key constants in [`src/App.jsx`](src/App.jsx):

| Constant | Default | Description |
|----------|---------|-------------|
| `GUIDE_FRAC` | `0.5` | Guide box size as a fraction of the frame |
| `GUIDE_OFFSET_X` | `0.12` | Horizontal offset of the guide box (screen right) |
| `PREDICT_INTERVAL_MS` | `350` | Milliseconds between inference runs |
| `STABILITY_WINDOW` | `5` | Frames considered for label smoothing |
| `STABILITY_REQUIRED` | `3` | Votes needed to change the displayed label |
| `HYSTERESIS` | `0.05` | Threshold band to reduce confident/unknown flicker |

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server with HMR |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Run ESLint |

## Browser notes

- **Camera:** rear-facing camera is preferred on mobile (`facingMode: 'environment'`).
- **Performance:** first inference after load may be slower while the WebGL backend warms up; a warmup pass runs on model load.
- **Bundle size:** TensorFlow.js is ~1.2 MB minified — expected for in-browser ML.

## License

Private project. Pokémon names and sprites are property of Nintendo / Game Freak / The Pokémon Company; used here for educational/demo purposes.
