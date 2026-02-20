import './style.css'
import { RealtimeVision, StreamClient, type StreamInferenceResult } from 'overshoot'

// ── Constants ──────────────────────────────────────────────────────────
const OVERSHOOT_API_KEY = import.meta.env.VITE_OVERSHOOT_API_KEY || ''
const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY || ''
const DEFAULT_OVERSHOOT_MODEL = 'Qwen/Qwen3-VL-32B-Instruct-FP8'
const GEMINI_WS_URL = 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'

// ── DOM Elements: Shared ───────────────────────────────────────────────
const videoInput = document.getElementById('video-input') as HTMLInputElement
const fileLabel = document.getElementById('file-label') as HTMLSpanElement
const promptInput = document.getElementById('prompt-input') as HTMLTextAreaElement
const startBtn = document.getElementById('start-btn') as HTMLButtonElement
const stopBtn = document.getElementById('stop-btn') as HTMLButtonElement
const copyBtn = document.getElementById('copy-btn') as HTMLButtonElement
const clearBtn = document.getElementById('clear-btn') as HTMLButtonElement
const videoPreview = document.getElementById('video-preview') as HTMLVideoElement
const videoPlaceholder = document.getElementById('video-placeholder') as HTMLDivElement
const statusDot = document.getElementById('status-dot') as HTMLSpanElement
const statusText = document.getElementById('status-text') as HTMLSpanElement
const resultsLog = document.getElementById('results-log') as HTMLDivElement
const resultsEmpty = document.getElementById('results-empty') as HTMLDivElement
const resultCount = document.getElementById('result-count') as HTMLSpanElement

// ── DOM Elements: Engine toggle ────────────────────────────────────────
const engineTabs = document.querySelectorAll('.engine-tab') as NodeListOf<HTMLButtonElement>
const overshootConfig = document.getElementById('overshoot-config') as HTMLDivElement
const geminiConfig = document.getElementById('gemini-config') as HTMLDivElement

// ── DOM Elements: Overshoot-specific ───────────────────────────────────
const modeSelect = document.getElementById('mode-select') as HTMLSelectElement
const clipOptions = document.getElementById('clip-options') as HTMLDivElement
const frameOptions = document.getElementById('frame-options') as HTMLDivElement
const clipLength = document.getElementById('clip-length') as HTMLInputElement
const clipDelay = document.getElementById('clip-delay') as HTMLInputElement
const clipFps = document.getElementById('clip-fps') as HTMLInputElement
const clipSampling = document.getElementById('clip-sampling') as HTMLInputElement
const frameInterval = document.getElementById('frame-interval') as HTMLInputElement
const modelSelect = document.getElementById('model-select') as HTMLSelectElement
const modelStatusHint = document.getElementById('model-status-hint') as HTMLSpanElement

// ── DOM Elements: Gemini-specific ──────────────────────────────────────
const geminiModelSelect = document.getElementById('gemini-model') as HTMLSelectElement
const geminiFpsInput = document.getElementById('gemini-fps') as HTMLInputElement

// ── Types ──────────────────────────────────────────────────────────────
interface ResultData {
  ok: boolean
  result: string
  error: string | null
  mode: string
  inference_latency_ms: number | null
  total_latency_ms: number | null
  finish_reason: string | null
}

interface TrackedResult {
  result: ResultData
  videoStartSec: number
  videoEndSec: number
}

type Engine = 'overshoot' | 'gemini'

// ── State ──────────────────────────────────────────────────────────────
let activeEngine: Engine = 'overshoot'
let selectedFile: File | null = null
let trackedResults: TrackedResult[] = []

let streamStartedAt = 0
let videoDurationSec = 0
let autoStopTimer: ReturnType<typeof setTimeout> | null = null

// Overshoot state
let vision: RealtimeVision | null = null

// Gemini state
let geminiWs: WebSocket | null = null
let geminiFrameInterval: ReturnType<typeof setInterval> | null = null
let geminiCanvas: HTMLCanvasElement | null = null
let geminiVideoEl: HTMLVideoElement | null = null
// Audio capture disabled for now — will revisit later
// let geminiAudioCtx: AudioContext | null = null
// let geminiAudioSource: MediaElementAudioSourceNode | null = null
// let geminiAudioProcessor: ScriptProcessorNode | null = null

// ── Engine toggle ──────────────────────────────────────────────────────
engineTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const engine = tab.dataset.engine as Engine
    if (engine === activeEngine) return
    activeEngine = engine

    engineTabs.forEach(t => t.classList.remove('active'))
    tab.classList.add('active')

    overshootConfig.style.display = engine === 'overshoot' ? '' : 'none'
    geminiConfig.style.display = engine === 'gemini' ? '' : 'none'
  })
})

// ── Overshoot model loading ────────────────────────────────────────────
const apiClient = new StreamClient({ apiKey: OVERSHOOT_API_KEY })

const MODEL_SIZE_ORDER: Record<string, number> = {
  '72B': 1, '32B': 2, '30B': 3, '8B': 4, '4B': 5, '2B': 6,
}

function getModelSortKey(model: string): number {
  for (const [size, order] of Object.entries(MODEL_SIZE_ORDER)) {
    if (model.includes(size)) return order
  }
  return 99
}

const STATUS_LABELS: Record<string, string> = {
  ready: '',
  degraded: ' (degraded)',
  saturated: ' (at capacity)',
  unavailable: ' (unavailable)',
}

async function loadModels() {
  modelSelect.innerHTML = '<option value="" disabled>Loading models...</option>'
  try {
    const models = await apiClient.getModels()
    models.sort((a, b) => {
      const aReady = a.ready ? 0 : 1
      const bReady = b.ready ? 0 : 1
      if (aReady !== bReady) return aReady - bReady
      return getModelSortKey(a.model) - getModelSortKey(b.model)
    })

    modelSelect.innerHTML = ''
    const readyGroup = document.createElement('optgroup')
    readyGroup.label = 'Available'
    const unavailGroup = document.createElement('optgroup')
    unavailGroup.label = 'Unavailable'

    for (const m of models) {
      const opt = document.createElement('option')
      opt.value = m.model
      const shortName = m.model.split('/').pop() || m.model
      opt.textContent = shortName + (STATUS_LABELS[m.status] || '')
      opt.disabled = !m.ready
      if (m.model === DEFAULT_OVERSHOOT_MODEL) opt.selected = true
      if (m.ready) readyGroup.appendChild(opt)
      else unavailGroup.appendChild(opt)
    }

    if (readyGroup.children.length) modelSelect.appendChild(readyGroup)
    if (unavailGroup.children.length) modelSelect.appendChild(unavailGroup)
    modelStatusHint.textContent = ''
  } catch (err) {
    console.warn('Failed to load models, using default', err)
    modelSelect.innerHTML = ''
    const opt = document.createElement('option')
    opt.value = DEFAULT_OVERSHOOT_MODEL
    opt.textContent = 'Qwen3-VL-32B-Instruct-FP8 (default)'
    opt.selected = true
    modelSelect.appendChild(opt)
    modelStatusHint.textContent = '(could not fetch live status)'
  }
}

loadModels()

// ── Overshoot mode toggle ──────────────────────────────────────────────
modeSelect.addEventListener('change', () => {
  const isClip = modeSelect.value === 'clip'
  clipOptions.style.display = isClip ? '' : 'none'
  frameOptions.style.display = isClip ? 'none' : ''
})

// ── File selection ─────────────────────────────────────────────────────
videoInput.addEventListener('change', () => {
  const file = videoInput.files?.[0]
  if (file) {
    selectedFile = file
    fileLabel.textContent = file.name
    fileLabel.classList.add('has-file')
    startBtn.disabled = false

    const url = URL.createObjectURL(file)
    videoPreview.src = url
    videoPreview.classList.add('active')
    videoPlaceholder.style.display = 'none'
    videoPreview.load()
    videoPreview.play().catch(() => {})

    videoPreview.addEventListener('loadedmetadata', () => {
      videoDurationSec = videoPreview.duration
    }, { once: true })
  }
})

// ── Start (dispatch) ───────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  if (!selectedFile) return

  setStatus('running')
  startBtn.disabled = true
  stopBtn.disabled = false
  lockControls(true)

  try {
    if (activeEngine === 'overshoot') {
      await startOvershoot()
    } else {
      await startGemini()
    }

    // Schedule auto-stop after single playback
    if (videoDurationSec > 0) {
      const bufferSec = activeEngine === 'overshoot'
        ? (parseFloat(clipLength.value) || 5) + 2
        : (parseFloat(geminiFpsInput.value) || 1) + 3
      const totalMs = videoDurationSec * 1000 + bufferSec * 1000

      autoStopTimer = setTimeout(async () => {
        if (activeEngine === 'overshoot' ? vision?.isActive() : geminiWs) {
          await stopProcessing(true)
        }
      }, totalMs)
    }
  } catch (err: any) {
    handleError(err)
    resetUI()
  }
})

// ── Stop (dispatch) ────────────────────────────────────────────────────
stopBtn.addEventListener('click', async () => {
  await stopProcessing(false)
})

async function stopProcessing(completed: boolean) {
  if (autoStopTimer) {
    clearTimeout(autoStopTimer)
    autoStopTimer = null
  }

  if (activeEngine === 'overshoot') {
    await stopOvershoot()
  } else {
    stopGeminiCleanup()
  }

  if (completed) {
    setStatus('done')
  } else {
    setStatus('idle')
  }
  stopBtn.disabled = true
  lockControls(false)
  startBtn.disabled = !selectedFile

  restoreVideoPreview()
}

// ═══════════════════════════════════════════════════════════════════════
//  OVERSHOOT ENGINE
// ═══════════════════════════════════════════════════════════════════════

async function startOvershoot() {
  const mode = modeSelect.value as 'clip' | 'frame'

  const config: ConstructorParameters<typeof RealtimeVision>[0] = {
    apiKey: OVERSHOOT_API_KEY,
    prompt: promptInput.value || 'Describe what you see',
    source: { type: 'video', file: selectedFile! },
    model: modelSelect.value,
    mode,
    onResult: handleOvershootResult,
    onError: handleError,
  }

  if (mode === 'clip') {
    config.clipProcessing = {
      clip_length_seconds: parseFloat(clipLength.value) || 5,
      delay_seconds: parseFloat(clipDelay.value) || 5,
      fps: parseInt(clipFps.value) || 30,
      sampling_ratio: parseFloat(clipSampling.value) || 1,
    }
  } else {
    config.frameProcessing = {
      interval_seconds: parseFloat(frameInterval.value) || 2,
    }
  }

  vision = new RealtimeVision(config)
  await vision.start()
  streamStartedAt = Date.now()

  const stream = vision.getMediaStream()
  if (stream) {
    videoPreview.srcObject = stream
    videoPreview.classList.add('active')
    videoPlaceholder.style.display = 'none'
    videoPreview.play().catch(() => {})
  }
}

async function stopOvershoot() {
  if (vision) {
    try { await vision.stop() } catch (_) {}
    vision = null
  }
}

function handleOvershootResult(result: StreamInferenceResult) {
  const now = Date.now()
  const elapsedMs = now - streamStartedAt
  const capturePointSec = Math.max(0, (elapsedMs - (result.total_latency_ms || 0)) / 1000)

  let videoStartSec: number
  let videoEndSec: number

  if (result.mode === 'clip') {
    const clipLen = parseFloat(clipLength.value) || 5
    videoStartSec = Math.max(0, capturePointSec - clipLen)
    videoEndSec = capturePointSec
  } else {
    videoStartSec = capturePointSec
    videoEndSec = capturePointSec
  }

  if (videoDurationSec > 0) {
    videoStartSec = Math.min(videoStartSec, videoDurationSec)
    videoEndSec = Math.min(videoEndSec, videoDurationSec)
  }

  const data: ResultData = {
    ok: result.ok,
    result: result.result,
    error: result.error,
    mode: result.mode,
    inference_latency_ms: result.inference_latency_ms,
    total_latency_ms: result.total_latency_ms,
    finish_reason: result.finish_reason,
  }

  trackedResults.push({ result: data, videoStartSec, videoEndSec })
  appendResultCard(trackedResults.length - 1)
}

// ═══════════════════════════════════════════════════════════════════════
//  GEMINI LIVE ENGINE
// ═══════════════════════════════════════════════════════════════════════

// ── Audio helpers (disabled — video-only mode for now) ──────────────────
//
// const GEMINI_AUDIO_SAMPLE_RATE = 16000
//
// function downsampleFloat32(buffer: Float32Array, fromRate: number, toRate: number): Float32Array {
//   if (fromRate === toRate) return buffer
//   const ratio = fromRate / toRate
//   const newLength = Math.round(buffer.length / ratio)
//   const result = new Float32Array(newLength)
//   for (let i = 0; i < newLength; i++) {
//     result[i] = buffer[Math.round(i * ratio)]
//   }
//   return result
// }
//
// function float32ToInt16(samples: Float32Array): Int16Array {
//   const out = new Int16Array(samples.length)
//   for (let i = 0; i < samples.length; i++) {
//     const s = Math.max(-1, Math.min(1, samples[i]))
//     out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
//   }
//   return out
// }
//
// function arrayBufferToBase64(buffer: ArrayBuffer): string {
//   const bytes = new Uint8Array(buffer)
//   let binary = ''
//   for (let i = 0; i < bytes.byteLength; i++) {
//     binary += String.fromCharCode(bytes[i])
//   }
//   return btoa(binary)
// }

// ── Gemini start / stop ─────────────────────────────────────────────────

async function startGemini() {
  if (!GEMINI_API_KEY) {
    throw new Error('Gemini API key is missing — set VITE_GEMINI_API_KEY in .env')
  }
  const apiKey = GEMINI_API_KEY

  const model = geminiModelSelect.value
  const frameIntervalSec = parseFloat(geminiFpsInput.value) || 1
  const prompt = promptInput.value || 'Describe what you see'
  // const needsAudio = model.includes('native-audio')

  const video = document.createElement('video')
  video.src = URL.createObjectURL(selectedFile!)
  video.muted = true
  video.playsInline = true
  video.crossOrigin = 'anonymous'

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve()
    video.onerror = () => reject(new Error('Failed to load video'))
    if (video.readyState >= 1) resolve()
  })

  await video.play()
  geminiVideoEl = video

  // Show the video in preview
  videoPreview.src = video.src
  videoPreview.classList.add('active')
  videoPlaceholder.style.display = 'none'
  videoPreview.load()
  videoPreview.play().catch(() => {})

  // Offscreen canvas for frame capture (768x768 optimal for Gemini)
  const canvas = document.createElement('canvas')
  const maxDim = 768
  const scale = Math.min(maxDim / video.videoWidth, maxDim / video.videoHeight, 1)
  canvas.width = Math.round(video.videoWidth * scale)
  canvas.height = Math.round(video.videoHeight * scale)
  geminiCanvas = canvas

  // Audio capture disabled — video-only mode for now
  // if (needsAudio) {
  //   const audioCtx = new AudioContext()
  //   geminiAudioCtx = audioCtx
  //   const source = audioCtx.createMediaElementSource(video)
  //   geminiAudioSource = source
  //   const processor = audioCtx.createScriptProcessor(4096, 1, 1)
  //   geminiAudioProcessor = processor
  //   processor.onaudioprocess = (e) => {
  //     if (!geminiWs || geminiWs.readyState !== WebSocket.OPEN) return
  //     const inputData = e.inputBuffer.getChannelData(0)
  //     const downsampled = downsampleFloat32(inputData, audioCtx.sampleRate, GEMINI_AUDIO_SAMPLE_RATE)
  //     const int16 = float32ToInt16(downsampled)
  //     const base64 = arrayBufferToBase64(int16.buffer)
  //     geminiWs.send(JSON.stringify({
  //       realtimeInput: {
  //         mediaChunks: [{ mimeType: `audio/pcm;rate=${GEMINI_AUDIO_SAMPLE_RATE}`, data: base64 }],
  //       },
  //     }))
  //   }
  //   source.connect(processor)
  //   const muteGain = audioCtx.createGain()
  //   muteGain.gain.value = 0
  //   processor.connect(muteGain)
  //   muteGain.connect(audioCtx.destination)
  // }

  // Open WebSocket
  const wsUrl = `${GEMINI_WS_URL}?key=${apiKey}`
  const ws = new WebSocket(wsUrl)
  geminiWs = ws

  ws.addEventListener('open', () => {
    ws.send(JSON.stringify({
      setup: {
        model: `models/${model}`,
        generationConfig: {
          responseModalities: ['AUDIO'],
        },
        outputAudioTranscription: {},
        systemInstruction: {
          parts: [{ text: prompt }],
        },
      },
    }))
  })

  let setupDone = false

  ws.addEventListener('message', (event) => {
    // Skip binary messages (audio data we don't need)
    if (typeof event.data !== 'string') return

    try {
      const msg = JSON.parse(event.data)
      console.log('[Gemini WS]', JSON.stringify(msg).slice(0, 200))

      if (msg.setupComplete) {
        setupDone = true
        streamStartedAt = Date.now()

        geminiFrameInterval = setInterval(() => {
          sendGeminiFrame(video, canvas, ws)
        }, frameIntervalSec * 1000)

        sendGeminiFrame(video, canvas, ws)
        return
      }

      if (msg.serverContent) {
        // Collect text from transcription or model turn parts
        let text = ''

        // Native audio model sends transcription separately
        const transcription = msg.serverContent.outputTranscription
        if (transcription?.text) {
          text += transcription.text
        }

        // Also check model turn parts for any text
        const parts = msg.serverContent.modelTurn?.parts || []
        for (const p of parts) {
          if (p.text) text += p.text
        }

        if (!text) return

        const now = Date.now()
        const elapsedMs = now - streamStartedAt
        const videoPosSec = Math.max(0, elapsedMs / 1000)

        let videoStartSec = Math.max(0, videoPosSec - frameIntervalSec)
        let videoEndSec = videoPosSec

        if (videoDurationSec > 0) {
          videoStartSec = Math.min(videoStartSec, videoDurationSec)
          videoEndSec = Math.min(videoEndSec, videoDurationSec)
        }

        const data: ResultData = {
          ok: true,
          result: text,
          error: null,
          mode: 'gemini',
          inference_latency_ms: null,
          total_latency_ms: null,
          finish_reason: null,
        }

        trackedResults.push({ result: data, videoStartSec, videoEndSec })
        appendResultCard(trackedResults.length - 1)
      }
    } catch (err) {
      console.warn('Failed to parse Gemini message:', err)
    }
  })

  ws.addEventListener('error', () => {
    handleError(new Error('Gemini WebSocket connection error'))
  })

  ws.addEventListener('close', (event) => {
    if (event.code !== 1000) {
      handleError(new Error(`Gemini connection closed: ${event.reason || 'unknown reason'} (code ${event.code})`))
    }
  })
}

function sendGeminiFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement, ws: WebSocket) {
  if (ws.readyState !== WebSocket.OPEN) return
  if (video.paused || video.ended) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

  canvas.toBlob((blob) => {
    if (!blob || ws.readyState !== WebSocket.OPEN) return

    const reader = new FileReader()
    reader.onloadend = () => {
      const base64 = (reader.result as string).split(',')[1]
      ws.send(JSON.stringify({
        realtimeInput: {
          mediaChunks: [{
            mimeType: 'image/jpeg',
            data: base64,
          }],
        },
      }))
    }
    reader.readAsDataURL(blob)
  }, 'image/jpeg', 0.8)
}

function stopGeminiCleanup() {
  if (geminiFrameInterval) {
    clearInterval(geminiFrameInterval)
    geminiFrameInterval = null
  }
  // Audio cleanup disabled — video-only mode for now
  // if (geminiAudioProcessor) {
  //   geminiAudioProcessor.disconnect()
  //   geminiAudioProcessor = null
  // }
  // if (geminiAudioSource) {
  //   geminiAudioSource.disconnect()
  //   geminiAudioSource = null
  // }
  // if (geminiAudioCtx) {
  //   geminiAudioCtx.close().catch(() => {})
  //   geminiAudioCtx = null
  // }
  if (geminiWs) {
    try { geminiWs.close(1000) } catch (_) {}
    geminiWs = null
  }
  if (geminiVideoEl) {
    geminiVideoEl.pause()
    URL.revokeObjectURL(geminiVideoEl.src)
    geminiVideoEl = null
  }
  geminiCanvas = null
}

// ═══════════════════════════════════════════════════════════════════════
//  SHARED: Copy, Clear, Results, UI Helpers
// ═══════════════════════════════════════════════════════════════════════

// ── Copy results ───────────────────────────────────────────────────────
copyBtn.addEventListener('click', async () => {
  const text = formatResultsForCopy()
  if (!text) {
    flashButton(copyBtn, 'Nothing to copy')
    return
  }
  try {
    await navigator.clipboard.writeText(text)
    flashButton(copyBtn, 'Copied!')
  } catch {
    flashButton(copyBtn, 'Copy failed')
  }
})

function formatResultsForCopy(): string {
  const meaningful = trackedResults.filter(t => {
    if (!t.result.ok) return false
    const text = t.result.result.trim()
    if (text === '--' || text === '-' || text === '') return false
    return true
  })

  if (meaningful.length === 0) return ''

  const sorted = [...meaningful].sort((a, b) => a.videoStartSec - b.videoStartSec)

  return sorted.map(t => {
    const timeLabel = `${fmtTime(t.videoStartSec)}–${fmtTime(t.videoEndSec)}`
    return `${timeLabel}\n${t.result.result.trim()}`
  }).join('\n\n')
}

function flashButton(btn: HTMLButtonElement, msg: string) {
  const original = btn.textContent
  btn.textContent = msg
  btn.disabled = true
  setTimeout(() => {
    btn.textContent = original
    btn.disabled = false
  }, 1500)
}

// ── Clear results ──────────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  trackedResults = []
  renderResultsFull()
})

// ── Error handler ──────────────────────────────────────────────────────
function handleError(error: Error) {
  console.error('Error:', error)
  setStatus('error')

  const el = document.createElement('div')
  el.className = 'error-banner'
  el.innerHTML = `<strong>${escapeHtml(error.name || 'Error')}:</strong> ${escapeHtml(error.message)}`

  if (resultsEmpty) resultsEmpty.style.display = 'none'
  resultsLog.prepend(el)
}

// ── Render results ─────────────────────────────────────────────────────
function renderResultsFull() {
  const count = trackedResults.length
  resultCount.textContent = `${count} result${count !== 1 ? 's' : ''}`

  if (count === 0) {
    resultsLog.innerHTML = ''
    resultsLog.appendChild(resultsEmpty)
    resultsEmpty.style.display = ''
    return
  }

  resultsEmpty.style.display = 'none'
  resultsLog.innerHTML = ''
  for (let i = trackedResults.length - 1; i >= 0; i--) {
    resultsLog.appendChild(createResultCard(trackedResults[i], i + 1))
  }
}

function appendResultCard(idx: number) {
  const count = trackedResults.length
  resultCount.textContent = `${count} result${count !== 1 ? 's' : ''}`
  resultsEmpty.style.display = 'none'

  const card = createResultCard(trackedResults[idx], idx + 1)
  resultsLog.prepend(card)
  resultsLog.scrollTop = 0
}

function createResultCard(tracked: TrackedResult, index: number): HTMLDivElement {
  const r = tracked.result
  const card = document.createElement('div')
  card.className = `result-card${r.ok ? '' : ' error'}`

  const timeLabel = `${fmtTime(tracked.videoStartSec)} – ${fmtTime(tracked.videoEndSec)}`

  const modeBadgeClass = r.mode === 'clip' ? 'mode-clip'
    : r.mode === 'frame' ? 'mode-frame'
    : 'mode-gemini'

  let metaHtml = `
    <span class="result-badge ${r.ok ? 'ok' : 'fail'}">${r.ok ? 'OK' : 'FAIL'}</span>
    <span class="result-badge ${modeBadgeClass}">${r.mode}</span>
    <span class="result-timestamp">#${index}</span>
    <span class="result-video-time">${timeLabel}</span>
  `

  if (r.inference_latency_ms != null) {
    metaHtml += `<span class="result-latency">inf ${Math.round(r.inference_latency_ms)}ms</span>`
  }
  if (r.total_latency_ms != null) {
    metaHtml += `<span class="result-latency">tot ${Math.round(r.total_latency_ms)}ms</span>`
  }

  let bodyHtml = ''
  if (r.ok) {
    bodyHtml = `<div class="result-text">${escapeHtml(r.result)}</div>`
  } else {
    bodyHtml = `<div class="result-error-text">${escapeHtml(r.error || 'Unknown error')}</div>`
  }

  if (r.finish_reason && r.finish_reason !== 'stop') {
    bodyHtml += `<div class="result-finish">finish_reason: ${escapeHtml(r.finish_reason)}</div>`
  }

  card.innerHTML = `
    <div class="result-meta">${metaHtml}</div>
    ${bodyHtml}
  `
  return card
}

// ── UI Helpers ─────────────────────────────────────────────────────────
function setStatus(state: 'idle' | 'running' | 'error' | 'done') {
  statusDot.className = 'status-dot'
  if (state === 'running') {
    statusDot.classList.add('running')
    statusText.textContent = 'Processing...'
  } else if (state === 'error') {
    statusDot.classList.add('error')
    statusText.textContent = 'Error'
  } else if (state === 'done') {
    statusDot.classList.add('done')
    statusText.textContent = 'Complete'
  } else {
    statusText.textContent = 'Idle'
  }
}

function lockControls(locked: boolean) {
  videoInput.disabled = locked
  promptInput.disabled = locked
  // Overshoot controls
  modeSelect.disabled = locked
  modelSelect.disabled = locked
  clipLength.disabled = locked
  clipDelay.disabled = locked
  clipFps.disabled = locked
  clipSampling.disabled = locked
  frameInterval.disabled = locked
  // Gemini controls
  geminiModelSelect.disabled = locked
  geminiFpsInput.disabled = locked
  // Engine tabs
  engineTabs.forEach(t => {
    if (locked) t.setAttribute('disabled', '')
    else t.removeAttribute('disabled')
  })
}

function resetUI() {
  setStatus('idle')
  startBtn.disabled = !selectedFile
  stopBtn.disabled = true
  lockControls(false)
  restoreVideoPreview()
}

function restoreVideoPreview() {
  if (selectedFile) {
    const url = URL.createObjectURL(selectedFile)
    videoPreview.srcObject = null
    videoPreview.src = url
    videoPreview.classList.add('active')
    videoPlaceholder.style.display = 'none'
    videoPreview.load()
  }
}

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function escapeHtml(str: string): string {
  const div = document.createElement('div')
  div.textContent = str
  return div.innerHTML
}
