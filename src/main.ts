import './style.css'
import { RealtimeVision, StreamClient, type StreamInferenceResult, type ModelInfo } from 'overshoot'

// ── Constants ──────────────────────────────────────────────────────────
const API_KEY = 'ovs_7c11ed0726d291fe01af7cf481bc84cf'
const DEFAULT_MODEL = 'Qwen/Qwen3-VL-32B-Instruct-FP8'

// ── DOM Elements ───────────────────────────────────────────────────────
const videoInput = document.getElementById('video-input') as HTMLInputElement
const fileLabel = document.getElementById('file-label') as HTMLSpanElement
const promptInput = document.getElementById('prompt-input') as HTMLTextAreaElement
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

// ── Types ──────────────────────────────────────────────────────────────
interface TrackedResult {
  result: StreamInferenceResult
  /** Estimated start of the clip/frame in source video (seconds) */
  videoStartSec: number
  /** Estimated end of the clip/frame in source video (seconds) */
  videoEndSec: number
}

// ── State ──────────────────────────────────────────────────────────────
let vision: RealtimeVision | null = null
let selectedFile: File | null = null
let trackedResults: TrackedResult[] = []

/** Wall-clock time (ms) when vision.start() resolved */
let streamStartedAt = 0
/** Duration of the source video in seconds (from the preview element) */
let videoDurationSec = 0
/** Timer ID for auto-stop after video finishes */
let autoStopTimer: ReturnType<typeof setTimeout> | null = null

// ── Model loading ──────────────────────────────────────────────────────
const apiClient = new StreamClient({ apiKey: API_KEY })

/** Known size tiers for sorting models sensibly */
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

    // Sort: ready first, then by size (large → small)
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
      if (m.model === DEFAULT_MODEL) opt.selected = true

      if (m.ready) {
        readyGroup.appendChild(opt)
      } else {
        unavailGroup.appendChild(opt)
      }
    }

    if (readyGroup.children.length) modelSelect.appendChild(readyGroup)
    if (unavailGroup.children.length) modelSelect.appendChild(unavailGroup)

    modelStatusHint.textContent = ''
  } catch (err) {
    console.warn('Failed to load models, using default', err)
    modelSelect.innerHTML = ''
    const opt = document.createElement('option')
    opt.value = DEFAULT_MODEL
    opt.textContent = 'Qwen3-VL-30B-A3B (default)'
    opt.selected = true
    modelSelect.appendChild(opt)
    modelStatusHint.textContent = '(could not fetch live status)'
  }
}

// Fire and forget on page load
loadModels()

// ── Mode toggle ────────────────────────────────────────────────────────
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

    // Show a local preview of the file and read its duration
    const url = URL.createObjectURL(file)
    videoPreview.src = url
    videoPreview.classList.add('active')
    videoPlaceholder.style.display = 'none'
    videoPreview.load()
    videoPreview.play().catch(() => {})

    // Capture duration once metadata loads
    videoPreview.addEventListener('loadedmetadata', () => {
      videoDurationSec = videoPreview.duration
    }, { once: true })
  }
})

// ── Start ──────────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  if (!selectedFile) return

  setStatus('running')
  startBtn.disabled = true
  stopBtn.disabled = false
  lockControls(true)

  const mode = modeSelect.value as 'clip' | 'frame'

  const config: ConstructorParameters<typeof RealtimeVision>[0] = {
    apiKey: API_KEY,
    prompt: promptInput.value || 'Describe what you see',
    source: { type: 'video', file: selectedFile },
    model: modelSelect.value,
    mode,
    onResult: handleResult,
    onError: handleError,
  }

  if (mode === 'clip') {
    config.clipProcessing = {
      clip_length_seconds: parseFloat(clipLength.value) || 1,
      delay_seconds: parseFloat(clipDelay.value) || 1,
      fps: parseInt(clipFps.value) || 30,
      sampling_ratio: parseFloat(clipSampling.value) || 0.1,
    }
  } else {
    config.frameProcessing = {
      interval_seconds: parseFloat(frameInterval.value) || 2,
    }
  }

  try {
    vision = new RealtimeVision(config)
    await vision.start()

    // Record start time for timestamp tracking
    streamStartedAt = Date.now()

    // Wire up the SDK media stream to the preview
    const stream = vision.getMediaStream()
    if (stream) {
      videoPreview.srcObject = stream
      videoPreview.classList.add('active')
      videoPlaceholder.style.display = 'none'
      videoPreview.play().catch(() => {})
    }

    // Schedule auto-stop when the video has played through once.
    // The SDK loops internally, so we stop it ourselves after one pass.
    // Add a buffer for the last clip to finish processing.
    if (videoDurationSec > 0) {
      const clipLenSec = mode === 'clip'
        ? (parseFloat(clipLength.value) || 1)
        : (parseFloat(frameInterval.value) || 2)
      const bufferMs = (clipLenSec + 2) * 1000  // clip length + 2s grace for last inference
      const totalMs = videoDurationSec * 1000 + bufferMs

      autoStopTimer = setTimeout(async () => {
        if (vision?.isActive()) {
          setStatus('done')
          await stopProcessing(true)
        }
      }, totalMs)
    }
  } catch (err: any) {
    handleError(err)
    resetUI()
  }
})

// ── Stop ───────────────────────────────────────────────────────────────
stopBtn.addEventListener('click', async () => {
  await stopProcessing(false)
})

async function stopProcessing(completed: boolean) {
  if (autoStopTimer) {
    clearTimeout(autoStopTimer)
    autoStopTimer = null
  }
  if (vision) {
    try {
      await vision.stop()
    } catch (_) {
      // ignore cleanup errors
    }
    vision = null
  }
  if (completed) {
    setStatus('done')
  } else {
    resetUI()
  }
  stopBtn.disabled = true
  lockControls(false)
  startBtn.disabled = !selectedFile

  // Restore local file preview
  if (selectedFile) {
    const url = URL.createObjectURL(selectedFile)
    videoPreview.srcObject = null
    videoPreview.src = url
    videoPreview.classList.add('active')
    videoPlaceholder.style.display = 'none'
    videoPreview.load()
  }
}

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
  // Filter to only successful, meaningful results (skip "--" and errors)
  const meaningful = trackedResults.filter(t => {
    if (!t.result.ok) return false
    const text = t.result.result.trim()
    if (text === '--' || text === '-' || text === '') return false
    return true
  })

  if (meaningful.length === 0) return ''

  // Sort chronologically (earliest first)
  const sorted = [...meaningful].sort((a, b) => a.videoStartSec - b.videoStartSec)

  return sorted.map(t => {
    const timeLabel = t.result.mode === 'clip'
      ? `${fmtTime(t.videoStartSec)}–${fmtTime(t.videoEndSec)}`
      : fmtTime(t.videoStartSec)

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

// ── Result handler ─────────────────────────────────────────────────────
function handleResult(result: StreamInferenceResult) {
  const now = Date.now()
  const elapsedMs = now - streamStartedAt

  // Estimate: the result was delivered now, but total_latency_ms ago the clip
  // was captured. So the video cursor at capture time was approximately:
  //   (elapsedMs - total_latency_ms) / 1000
  // For clip mode the clip covers [cursor - clip_length, cursor].
  // For frame mode it's a single point in time.
  const capturePointSec = Math.max(0, (elapsedMs - (result.total_latency_ms || 0)) / 1000)

  let videoStartSec: number
  let videoEndSec: number

  if (result.mode === 'clip') {
    const clipLen = parseFloat(clipLength.value) || 1
    videoStartSec = Math.max(0, capturePointSec - clipLen)
    videoEndSec = capturePointSec
  } else {
    videoStartSec = capturePointSec
    videoEndSec = capturePointSec
  }

  // Clamp to video duration
  if (videoDurationSec > 0) {
    videoStartSec = Math.min(videoStartSec, videoDurationSec)
    videoEndSec = Math.min(videoEndSec, videoDurationSec)
  }

  trackedResults.push({ result, videoStartSec, videoEndSec })
  appendResultCard(trackedResults.length - 1)
}

// ── Error handler ──────────────────────────────────────────────────────
function handleError(error: Error) {
  console.error('Overshoot error:', error)
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
  // Newest first
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

  // Video timecode
  const timeLabel = r.mode === 'clip'
    ? `${fmtTime(tracked.videoStartSec)} – ${fmtTime(tracked.videoEndSec)}`
    : fmtTime(tracked.videoStartSec)

  let metaHtml = `
    <span class="result-badge ${r.ok ? 'ok' : 'fail'}">${r.ok ? 'OK' : 'FAIL'}</span>
    <span class="result-badge mode-${r.mode}">${r.mode}</span>
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
  modeSelect.disabled = locked
  modelSelect.disabled = locked
  clipLength.disabled = locked
  clipDelay.disabled = locked
  clipFps.disabled = locked
  clipSampling.disabled = locked
  frameInterval.disabled = locked
}

function resetUI() {
  setStatus('idle')
  startBtn.disabled = !selectedFile
  stopBtn.disabled = true
  lockControls(false)

  // Restore local file preview if we have one
  if (selectedFile) {
    const url = URL.createObjectURL(selectedFile)
    videoPreview.srcObject = null
    videoPreview.src = url
    videoPreview.classList.add('active')
    videoPlaceholder.style.display = 'none'
    videoPreview.load()
  }
}

/** Format seconds as MM:SS */
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
