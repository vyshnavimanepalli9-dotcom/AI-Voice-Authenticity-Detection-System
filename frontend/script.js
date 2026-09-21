/* ==========================================================================
   AI VOICE AUTHENTICITY DETECTION AGENT - JAVASCRIPT LOGIC
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let currentAudioBlob = null;
    let currentAudioFile = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordTimerInterval = null;
    let recordSeconds = 0;
    
    // Web Audio API WAV Recorder State
    let audioCtx = null;
    let mediaStream = null;
    let scriptProcessor = null;
    let pcmBuffers = [];
    let recordingSampleRate = 44100;
    let isWavRecording = false;

    const API_BASE_URL = '/api';


    // --- DOM Element References ---
    const themeToggleBtn = document.getElementById('theme-toggle');
    const modeBadge = document.getElementById('mode-badge');
    const modeText = document.getElementById('mode-text');
    
    // Tabs & Panels
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Upload Elements
    const dropZone = document.getElementById('drop-zone');
    const audioFileInput = document.getElementById('audio-file-input');
    
    // Record Elements
    const startRecordBtn = document.getElementById('start-record-btn');
    const stopRecordBtn = document.getElementById('stop-record-btn');
    const recordIndicator = document.getElementById('record-indicator');
    const recordTimer = document.getElementById('record-timer');
    const recordStatusText = document.getElementById('record-status-text');

    // Preview Elements
    const audioPreviewCard = document.getElementById('audio-preview-card');
    const fileNameDisplay = document.getElementById('file-name-display');
    const clearAudioBtn = document.getElementById('clear-audio-btn');
    const audioPlayer = document.getElementById('audio-player');
    const waveformCanvas = document.getElementById('waveform-canvas');
    const analyzeBtn = document.getElementById('analyze-btn');

    // Loading & Output Dashboard
    const loadingOverlay = document.getElementById('loading-overlay');
    const progressFill = document.getElementById('progress-fill');
    const emptyState = document.getElementById('empty-state');
    const resultsCard = document.getElementById('results-card');

    // Verdict Elements
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictIconI = document.getElementById('verdict-icon-i');
    const predictionTitle = document.getElementById('prediction-title');
    const calibratedStatusTag = document.getElementById('calibrated-status-tag');
    const confidenceVal = document.getElementById('confidence-val');
    const demoNoticeBox = document.getElementById('demo-notice-box');
    const demoNoticeText = document.getElementById('demo-notice-text');

    // Metrics & Features
    const mDuration = document.getElementById('m-duration');
    const mSr = document.getElementById('m-sr');
    const mTime = document.getElementById('m-time');
    const mModel = document.getElementById('m-model');

    const fPitch = document.getElementById('f-pitch');
    const fPitchVar = document.getElementById('f-pitch-var');
    const fCentroid = document.getElementById('f-centroid');
    const fRolloff = document.getElementById('f-rolloff');
    const fZcr = document.getElementById('f-zcr');
    const fMfccVar = document.getElementById('f-mfcc-var');

    // XAI
    const xaiSummaryText = document.getElementById('xai-summary-text');
    const xaiFactorsContainer = document.getElementById('xai-factors-container');

    // --- 1. System Status & Model Info Fetch ---
    fetchModelStatus();

    async function fetchModelStatus() {
        try {
            const res = await fetch(`${API_BASE_URL}/model-info`);
            if (!res.ok) throw new Error("Status check failed");
            const data = await res.json();

            if (data.mode === 'TRAINED MODEL MODE') {
                modeBadge.className = 'badge badge-trained';
                modeText.textContent = 'TRAINED MODEL MODE (Calibrated)';
            } else {
                modeBadge.className = 'badge badge-demo';
                modeText.textContent = 'DEMO MODE (Uncalibrated)';
            }
        } catch (err) {
            console.warn("Could not connect to model status endpoint:", err);
            modeBadge.className = 'badge badge-demo';
            modeText.textContent = 'OFFLINE / DEMO MODE';
        }
    }

    // --- 2. Theme Switching ---
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        const icon = themeToggleBtn.querySelector('i');
        icon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    }

    // --- 3. Tab Switching ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetTab = document.getElementById(btn.dataset.tab);
            if (targetTab) targetTab.classList.add('active');
        });
    });

    // --- 4. Drag and Drop & File Input ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        // Validate extension
        const allowedExts = ['wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'];
        const ext = file.name.split('.').pop().toLowerCase();
        if (!allowedExts.includes(ext)) {
            alert(`Unsupported audio format (.${ext}). Please select a WAV, MP3, OGG, FLAC, M4A, or WEBM file.`);
            return;
        }

        // Validate size (15 MB)
        if (file.size > 15 * 1024 * 1024) {
            alert(`File size exceeds 15 MB limit. Selected file is ${round(file.size / (1024 * 1024), 2)} MB.`);
            return;
        }

        currentAudioFile = file;
        currentAudioBlob = file;
        fileNameDisplay.innerHTML = `<i class="fa-solid fa-file-audio"></i> ${file.name}`;

        const audioUrl = URL.createObjectURL(file);
        audioPlayer.src = audioUrl;

        audioPreviewCard.classList.remove('hidden');
        drawDummyWaveform();
    }

    // --- 5. Live Audio Recording (Web Audio API WAV Capture) ---
    startRecordBtn.addEventListener('click', async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Microphone recording is not supported in your browser.");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaStream = stream;

            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            recordingSampleRate = audioCtx.sampleRate;
            
            const source = audioCtx.createMediaStreamSource(stream);
            scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
            pcmBuffers = [];
            isWavRecording = true;

            scriptProcessor.onaudioprocess = (e) => {
                if (!isWavRecording) return;
                const inputBuffer = e.inputBuffer.getChannelData(0);
                pcmBuffers.push(new Float32Array(inputBuffer));
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(audioCtx.destination);

            recordSeconds = 0;
            recordTimer.textContent = "00:00";
            recordIndicator.classList.add('recording');
            recordStatusText.textContent = "Recording live audio... speak into microphone";

            startRecordBtn.disabled = true;
            stopRecordBtn.disabled = false;

            recordTimerInterval = setInterval(() => {
                recordSeconds++;
                const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                const secs = String(recordSeconds % 60).padStart(2, '0');
                recordTimer.textContent = `${mins}:${secs}`;

                if (recordSeconds >= 60) {
                    stopRecording();
                }
            }, 1000);

        } catch (err) {
            alert(`Microphone access error: ${err.message}`);
        }
    });

    stopRecordBtn.addEventListener('click', stopRecording);

    function stopRecording() {
        isWavRecording = false;
        clearInterval(recordTimerInterval);

        if (scriptProcessor) {
            try { scriptProcessor.disconnect(); } catch (e) {}
        }
        if (audioCtx && audioCtx.state !== 'closed') {
            try { audioCtx.close(); } catch (e) {}
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }

        recordIndicator.classList.remove('recording');
        recordStatusText.textContent = "Recording captured successfully!";
        startRecordBtn.disabled = false;
        stopRecordBtn.disabled = true;

        if (pcmBuffers.length === 0) {
            alert("No audio data was captured. Please check your microphone.");
            return;
        }

        // Encode captured PCM float buffers into standard 16-bit PCM WAV Blob
        const wavBlob = encodeWAV(pcmBuffers, recordingSampleRate);
        currentAudioBlob = wavBlob;
        currentAudioFile = new File([wavBlob], `recorded_voice_${Date.now()}.wav`, { type: 'audio/wav' });

        const audioUrl = URL.createObjectURL(wavBlob);
        audioPlayer.src = audioUrl;

        fileNameDisplay.innerHTML = `<i class="fa-solid fa-microphone"></i> Recorded_Voice.wav`;
        audioPreviewCard.classList.remove('hidden');
        drawDummyWaveform();
    }

    function encodeWAV(samplesBlocks, sampleRate) {
        let totalLength = 0;
        for (let b of samplesBlocks) totalLength += b.length;

        let buffer = new ArrayBuffer(44 + totalLength * 2);
        let view = new DataView(buffer);

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        /* RIFF identifier */
        writeString(view, 0, 'RIFF');
        /* RIFF chunk length */
        view.setUint32(4, 36 + totalLength * 2, true);
        /* RIFF format */
        writeString(view, 8, 'WAVE');
        /* Subchunk1 ID: "fmt " */
        writeString(view, 12, 'fmt ');
        /* Subchunk1 size: 16 */
        view.setUint32(16, 16, true);
        /* Audio format: 1 (PCM) */
        view.setUint16(20, 1, true);
        /* Num channels: 1 (Mono) */
        view.setUint16(22, 1, true);
        /* Sample rate */
        view.setUint32(24, sampleRate, true);
        /* Byte rate */
        view.setUint32(28, sampleRate * 2, true);
        /* Block align */
        view.setUint16(32, 2, true);
        /* Bits per sample: 16 */
        view.setUint16(34, 16, true);
        /* Subchunk2 ID: "data" */
        writeString(view, 36, 'data');
        /* Subchunk2 size */
        view.setUint32(40, totalLength * 2, true);

        /* Write 16-bit PCM samples */
        let offset = 44;
        for (let block of samplesBlocks) {
            for (let i = 0; i < block.length; i++) {
                let s = Math.max(-1, Math.min(1, block[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([view], { type: 'audio/wav' });
    }


    // --- 6. Clear Preview ---
    clearAudioBtn.addEventListener('click', () => {
        currentAudioFile = null;
        currentAudioBlob = null;
        audioPlayer.src = '';
        audioFileInput.value = '';
        audioPreviewCard.classList.add('hidden');
    });

    // --- 7. Analyze Voice Authenticity Request ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentAudioBlob) {
            alert("Please select or record an audio file first.");
            return;
        }

        // Show loading state
        audioPreviewCard.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');
        animateProgressBar();

        const formData = new FormData();
        formData.append('audio', currentAudioFile || currentAudioBlob);

        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loadingOverlay.classList.add('hidden');
            audioPreviewCard.classList.remove('hidden');

            if (!response.ok || !data.success) {
                alert(`Analysis Error: ${data.error || 'Failed to process audio file.'}`);
                return;
            }

            // Render Results Dashboard
            renderResults(data);

        } catch (err) {
            loadingOverlay.classList.add('hidden');
            audioPreviewCard.classList.remove('hidden');
            alert(`Network / Connection Error: ${err.message}`);
        }
    });

    function animateProgressBar() {
        progressFill.style.width = '0%';
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 90) {
                clearInterval(interval);
            } else {
                width += 15;
                progressFill.style.width = `${width}%`;
            }
        }, 200);
    }

    // --- 8. Render Results Dashboard ---
    function renderResults(data) {
        emptyState.classList.add('hidden');
        resultsCard.classList.remove('hidden');

        const isHuman = data.is_human;

        // Verdict Banner styling
        if (isHuman) {
            verdictBanner.className = 'verdict-banner verdict-human';
            verdictIconI.className = 'fa-solid fa-user-check';
            predictionTitle.textContent = 'REAL HUMAN VOICE';
            predictionTitle.style.color = 'var(--color-human)';
        } else {
            verdictBanner.className = 'verdict-banner verdict-ai';
            verdictIconI.className = 'fa-solid fa-robot';
            predictionTitle.textContent = 'AI-GENERATED VOICE';
            predictionTitle.style.color = 'var(--color-ai)';
        }

        confidenceVal.textContent = `${data.confidence}%`;

        if (data.demo_mode) {
            calibratedStatusTag.textContent = 'Uncalibrated (Demo Heuristics)';
            demoNoticeBox.classList.remove('hidden');
            demoNoticeText.textContent = data.status_note;
        } else {
            calibratedStatusTag.textContent = 'Calibrated ML Prediction';
            demoNoticeBox.classList.add('hidden');
        }

        // Metrics Grid
        mDuration.textContent = `${data.audio_metrics.duration_seconds} s`;
        mSr.textContent = `${data.audio_metrics.sample_rate_hz} Hz`;
        mTime.textContent = `${data.processing_time_ms} ms`;
        mModel.textContent = data.model_used;

        // Feature Summary Grid
        const feat = data.feature_summary;
        fPitch.textContent = `${feat.pitch_f0_mean_hz} Hz`;
        fPitchVar.textContent = feat.pitch_variation_index;
        fCentroid.textContent = `${feat.spectral_centroid_hz} Hz`;
        fRolloff.textContent = `${feat.spectral_rolloff_hz} Hz`;
        fZcr.textContent = feat.zero_crossing_rate;
        fMfccVar.textContent = feat.mfcc_variance;

        // Draw waveform from backend data if available
        if (data.waveform_data) {
            drawCustomWaveform(data.waveform_data);
        }

        // Render Explainable AI (XAI) Breakdown
        const xai = data.xai_explanation;
        xaiSummaryText.textContent = xai.summary;

        xaiFactorsContainer.innerHTML = '';
        xai.factors.forEach(factor => {
            const card = document.createElement('div');
            card.className = 'xai-factor-card';
            card.innerHTML = `
                <div class="xai-title">${factor.feature}</div>
                <div class="xai-obs">${factor.observation}</div>
                <div class="xai-impact">${factor.impact}</div>
            `;
            xaiFactorsContainer.appendChild(card);
        });

        // Scroll gracefully to results panel
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    // --- 9. Waveform Visualizer Canvas Drawing ---
    function drawDummyWaveform() {
        const points = [];
        for (let i = 0; i < 80; i++) {
            points.push(Math.random() * 0.7 + 0.1);
        }
        drawCustomWaveform(points);
    }

    function drawCustomWaveform(points) {
        const ctx = waveformCanvas.getContext('2d');
        const width = waveformCanvas.width = waveformCanvas.offsetWidth;
        const height = waveformCanvas.height = 80;

        ctx.clearRect(0, 0, width, height);

        const barWidth = width / points.length;
        const center = height / 2;

        ctx.fillStyle = '#38bdf8';

        points.forEach((val, index) => {
            const barHeight = val * (height * 0.8);
            const x = index * barWidth;
            const y = center - barHeight / 2;
            
            ctx.beginPath();
            ctx.roundRect(x + 1, y, Math.max(1, barWidth - 2), barHeight, 2);
            ctx.fill();
        });
    }

    function round(num, decimals) {
        return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
    }
});
