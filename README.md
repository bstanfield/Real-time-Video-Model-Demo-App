# Real-time Video Model Demo App

A browser-based demo app that uploads video files to the [Overshoot](https://docs.overshoot.ai/) real-time vision API and displays AI inference results as they stream back.

Built for UX research workflows -- upload a usability test screen recording, and the AI analyzes each clip for interesting user behavior, flagging moments worth probing with participants.

## How It Works

1. **Upload a video** -- Select any video file from your machine.
2. **Configure processing** -- Choose clip vs. frame mode, adjust clip length, delay, FPS, sampling ratio, and select from available models (fetched live from the Overshoot API).
3. **Start processing** -- The Overshoot SDK plays the video in-browser, streams frames over WebRTC to Overshoot's servers, and returns AI inference results via WebSocket.
4. **View results in real-time** -- Each result card shows an estimated source video timestamp, the model's response, latency metrics, and status.
5. **Auto-stop** -- Processing stops automatically after the video plays through once (no looping).
6. **Copy results** -- Click "Copy" to export all meaningful results to clipboard in chronological order, formatted with video timestamps.

## Video Timestamp Tracking

Overshoot's API doesn't include source video timestamps in results. This app builds a client-side harness that records wall-clock time when processing starts, then estimates each result's position in the source video by subtracting `total_latency_ms` from elapsed time. For clip mode, timestamps show the range the clip covers (e.g., `1:23–1:28`).

## Tech Stack

- **[Vite](https://vite.dev/)** -- dev server and bundler
- **TypeScript** -- type-safe integration with the Overshoot SDK
- **Vanilla HTML/CSS** -- no framework, minimal footprint
- **[Overshoot SDK](https://docs.overshoot.ai/)** (`overshoot` npm package) -- handles WebRTC streaming, frame capture, and result delivery

## Setup

```bash
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## Configuration

The Overshoot API key is hardcoded in `src/main.ts`. Replace it with your own key from [platform.overshoot.ai](https://platform.overshoot.ai/api-keys).

## Build

```bash
npm run build
npm run preview
```
