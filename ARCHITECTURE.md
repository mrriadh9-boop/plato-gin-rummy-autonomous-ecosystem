# Architecture Specification: Plato Gin Rummy Autonomous Ecosystem

## 1. Perception Pipeline (Dual-Stage Vision)
* **Classical Preprocessing (`vision/pipeline/roi_slicer.py`)**:
  - Sobel vertical gradient projection locates card centers dynamically without relying on fixed pitch.
  - Slices normalized 48x48 RGB float32 corner index glyphs.
  - Adaptive contrast normalization for table illumination variations.
* **UltraFastCardNet (`vision/classifier/ultra_fast_card_net.py`)**:
  - 52-class Convolutional Neural Network exported to ONNX format (`vision/classifier/models/ultra_fast_card_net.onnx`).
  - Measures 0.57ms latency per card on CPU.
* **State Aggregation & Reconciliation (`vision/pipeline/state_aggregator.py`)**:
  - 1D Hungarian tracker associates card tracks across frames.
  - Bayesian posterior history smoothing ($P_t(c) \propto P_{t-1}(c)^\gamma \cdot P_{\text{obs}}(c)$).
  - 52-card conservation reconciliation: enforces uniqueness via global Hungarian assignment with hard discard masking.

## 2. Neural Policy & IS-MCTS Decision Engine
* **Observation Representation**:
  - $8 \times 4 \times 13$ Spatial Planes: Player hand, discard top, discard history, opponent known picks, opponent known discards, Discard-Lock mask, formed melds, deadwood.
  - 16-dimensional Context Vector: Match score delta, stock remaining, turn number, deadwood points, pass counter.
* **Recurrent Policy Network (`ai_engine/models/recurrent_net.py`)**:
  - Dual-stream convolutions (Sets stream: $4 \times 1$ convs, Runs stream: $1 \times 13$ convs).
  - 256-dim GRU cell with MaskedCategorical action head (110 discrete moves).
* **Belief Supervised Head**:
  - 52-dimensional Bernoulli logit output predicting opponent private cards.
  - Deconflicted during training via PCGrad orthogonal projection.

## 3. Hardware-in-the-Loop Driver & Desktop Cockpit
* **Scrcpy Stream Client (`driver/capture/scrcpy_client.py`)**:
  - Direct TCP demuxing of H.264 stream via PyAV frame buffer.
  - Flags: `--turn-screen-off --stay-awake` for battery conservation.
* **Touch Dispatcher (`driver/dispatcher/touch_dispatcher.py`)**:
  - 0px coordinate translation on 1800x2880 Xiaomi Pad 6 canvas.
  - Timing debouncing prevents double-taps.
* **Production UI (`driver/hud/production_app.py`)**:
  - PyQt6 dark theme desktop app with live video canvas, card overlays, opponent belief heatmap, win rate meter, and telemetry graphs.
