/* ==========================================================================
   AI VOICE AUTHENTICITY DETECTION AGENT
   FRONTEND JAVASCRIPT
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {

    /* ======================================================================
       1. APPLICATION STATE
       ====================================================================== */

    let currentAudioBlob = null;
    let currentAudioFile = null;

    let mediaStream = null;
    let audioCtx = null;
    let scriptProcessor = null;

    let pcmBuffers = [];
    let recordingSampleRate = 44100;

    let recordTimerInterval = null;
    let recordSeconds = 0;
    let isWavRecording = false;
    let isAnalyzing = false;

    const API_BASE_URL = '/api';


    /* ======================================================================
       2. DOM REFERENCES
       ====================================================================== */

    // Theme
    const themeToggleBtn = document.getElementById('theme-toggle');

    // Model status
    const modeBadge = document.getElementById('mode-badge');
    const modeText = document.getElementById('mode-text');

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Upload
    const dropZone = document.getElementById('drop-zone');
    const audioFileInput = document.getElementById('audio-file-input');

    // Recording
    const startRecordBtn = document.getElementById('start-record-btn');
    const stopRecordBtn = document.getElementById('stop-record-btn');
    const recordIndicator = document.getElementById('record-indicator');
    const recordTimer = document.getElementById('record-timer');
    const recordStatusText = document.getElementById('record-status-text');

    // Preview
    const audioPreviewCard = document.getElementById('audio-preview-card');
    const fileNameDisplay = document.getElementById('file-name-display');
    const clearAudioBtn = document.getElementById('clear-audio-btn');
    const audioPlayer = document.getElementById('audio-player');
    const waveformCanvas = document.getElementById('waveform-canvas');
    const analyzeBtn = document.getElementById('analyze-btn');

    // Loading
    const loadingOverlay = document.getElementById('loading-overlay');
    const progressFill = document.getElementById('progress-fill');

    // Results
    const emptyState = document.getElementById('empty-state');
    const resultsCard = document.getElementById('results-card');

    // Verdict
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictIconI = document.getElementById('verdict-icon-i');
    const predictionTitle = document.getElementById('prediction-title');
    const calibratedStatusTag = document.getElementById('calibrated-status-tag');
    const confidenceVal = document.getElementById('confidence-val');

    const demoNoticeBox = document.getElementById('demo-notice-box');
    const demoNoticeText = document.getElementById('demo-notice-text');

    // Metrics
    const mDuration = document.getElementById('m-duration');
    const mSr = document.getElementById('m-sr');
    const mTime = document.getElementById('m-time');
    const mModel = document.getElementById('m-model');

    // Features
    const fPitch = document.getElementById('f-pitch');
    const fPitchVar = document.getElementById('f-pitch-var');
    const fCentroid = document.getElementById('f-centroid');
    const fRolloff = document.getElementById('f-rolloff');
    const fZcr = document.getElementById('f-zcr');
    const fMfccVar = document.getElementById('f-mfcc-var');

    // XAI
    const xaiSummaryText = document.getElementById('xai-summary-text');
    const xaiFactorsContainer = document.getElementById('xai-factors-container');


    /* ======================================================================
       3. INITIALIZATION
       ====================================================================== */

    initialize();

    function initialize() {
        initializeTheme();
        initializeTabs();
        initializeDragAndDrop();
        initializeRecording();
        initializeUpload();
        initializeAnalysis();
        initializeClearButton();

        fetchModelStatus();

        console.log(
            '%cVoiceAuthentic.AI initialized',
            'font-weight:bold; font-size:14px;'
        );
    }


    /* ======================================================================
       4. MODEL STATUS
       ====================================================================== */

    async function fetchModelStatus() {

        try {

            const response = await fetch(
                `${API_BASE_URL}/model-info`,
                {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json'
                    },
                    cache: 'no-cache'
                }
            );

            const contentType =
                response.headers.get('content-type') || '';

            if (!response.ok) {
                throw new Error(
                    `Model status request failed with HTTP ${response.status}`
                );
            }

            if (!contentType.includes('application/json')) {
                throw new Error(
                    'Model status endpoint returned a non-JSON response.'
                );
            }

            const data = await response.json();

            updateModelStatus(data);

        } catch (error) {

            console.warn(
                'Could not connect to model status endpoint:',
                error
            );

            setOfflineStatus();
        }
    }


    function updateModelStatus(data) {

        if (!modeBadge || !modeText) {
            return;
        }

        const isTrained =
            data &&
            (
                data.mode === 'TRAINED MODEL MODE' ||
                data.status === 'active'
            );

        if (isTrained) {

            modeBadge.className = 'badge badge-trained';

            modeText.textContent =
                'TRAINED MODEL MODE';

        } else {

            modeBadge.className = 'badge badge-demo';

            modeText.textContent =
                'DEMO / UNCALIBRATED MODE';
        }
    }


    function setOfflineStatus() {

        if (!modeBadge || !modeText) {
            return;
        }

        modeBadge.className = 'badge badge-demo';

        modeText.textContent =
            'API STATUS UNKNOWN';
    }


    /* ======================================================================
       5. THEME
       ====================================================================== */

    function initializeTheme() {

        if (!themeToggleBtn) {
            return;
        }

        const savedTheme =
            localStorage.getItem('theme') || 'dark';

        document.documentElement.setAttribute(
            'data-theme',
            savedTheme
        );

        updateThemeIcon(savedTheme);

        themeToggleBtn.addEventListener(
            'click',
            () => {

                const currentTheme =
                    document.documentElement.getAttribute(
                        'data-theme'
                    );

                const newTheme =
                    currentTheme === 'dark'
                        ? 'light'
                        : 'dark';

                document.documentElement.setAttribute(
                    'data-theme',
                    newTheme
                );

                localStorage.setItem(
                    'theme',
                    newTheme
                );

                updateThemeIcon(newTheme);
            }
        );
    }


    function updateThemeIcon(theme) {

        if (!themeToggleBtn) {
            return;
        }

        const icon =
            themeToggleBtn.querySelector('i');

        if (!icon) {
            return;
        }

        icon.className =
            theme === 'dark'
                ? 'fa-solid fa-sun'
                : 'fa-solid fa-moon';
    }


    /* ======================================================================
       6. TAB SWITCHING
       ====================================================================== */

    function initializeTabs() {

        tabBtns.forEach(button => {

            button.addEventListener(
                'click',
                () => {

                    tabBtns.forEach(btn => {
                        btn.classList.remove('active');
                    });

                    tabContents.forEach(content => {
                        content.classList.remove('active');
                    });

                    button.classList.add('active');

                    const targetId =
                        button.dataset.tab;

                    const target =
                        document.getElementById(targetId);

                    if (target) {
                        target.classList.add('active');
                    }
                }
            );
        });
    }


    /* ======================================================================
       7. DRAG & DROP
       ====================================================================== */

    function initializeDragAndDrop() {

        if (!dropZone) {
            return;
        }

        [
            'dragenter',
            'dragover',
            'dragleave',
            'drop'
        ].forEach(eventName => {

            dropZone.addEventListener(
                eventName,
                preventDefaults,
                false
            );
        });


        [
            'dragenter',
            'dragover'
        ].forEach(eventName => {

            dropZone.addEventListener(
                eventName,
                () => {
                    dropZone.classList.add('dragover');
                },
                false
            );
        });


        [
            'dragleave',
            'drop'
        ].forEach(eventName => {

            dropZone.addEventListener(
                eventName,
                () => {
                    dropZone.classList.remove('dragover');
                },
                false
            );
        });


        dropZone.addEventListener(
            'drop',
            event => {

                const files =
                    event.dataTransfer.files;

                if (files && files.length > 0) {

                    handleSelectedFile(
                        files[0]
                    );
                }
            }
        );
    }


    function preventDefaults(event) {

        event.preventDefault();
        event.stopPropagation();
    }


    /* ======================================================================
       8. FILE UPLOAD
       ====================================================================== */

    function initializeUpload() {

        if (!audioFileInput) {
            return;
        }

        audioFileInput.addEventListener(
            'change',
            event => {

                const files =
                    event.target.files;

                if (files && files.length > 0) {

                    handleSelectedFile(
                        files[0]
                    );
                }
            }
        );
    }


    function handleSelectedFile(file) {

        if (!file) {
            return;
        }


        /* --------------------------------------------------------------
           Validate extension
        -------------------------------------------------------------- */

        const allowedExtensions = [
            'wav',
            'mp3',
            'ogg',
            'flac',
            'm4a',
            'webm'
        ];

        const extension =
            file.name
                .split('.')
                .pop()
                .toLowerCase();


        if (!allowedExtensions.includes(extension)) {

            alert(
                `Unsupported audio format (.${extension}).\n\n` +
                `Supported formats: WAV, MP3, OGG, FLAC, M4A, WEBM.`
            );

            return;
        }


        /* --------------------------------------------------------------
           Validate size
        -------------------------------------------------------------- */

        const maxSize =
            15 * 1024 * 1024;


        if (file.size > maxSize) {

            alert(
                `File size exceeds the 15 MB limit.\n\n` +
                `Selected file: ${round(
                    file.size / (1024 * 1024),
                    2
                )} MB`
            );

            return;
        }


        /* --------------------------------------------------------------
           Stop any existing recording
        -------------------------------------------------------------- */

        if (isWavRecording) {
            stopRecording();
        }


        /* --------------------------------------------------------------
           Store file
        -------------------------------------------------------------- */

        currentAudioFile = file;
        currentAudioBlob = file;


        /* --------------------------------------------------------------
           Display filename
        -------------------------------------------------------------- */

        if (fileNameDisplay) {

            fileNameDisplay.innerHTML =
                `<i class="fa-solid fa-file-audio"></i> ` +
                escapeHtml(file.name);
        }


        /* --------------------------------------------------------------
           Audio preview
        -------------------------------------------------------------- */

        if (audioPlayer) {

            if (audioPlayer.src) {
                URL.revokeObjectURL(
                    audioPlayer.src
                );
            }

            const audioUrl =
                URL.createObjectURL(file);

            audioPlayer.src = audioUrl;
        }


        /* --------------------------------------------------------------
           Show preview
        -------------------------------------------------------------- */

        if (audioPreviewCard) {
            audioPreviewCard.classList.remove('hidden');
        }


        /* --------------------------------------------------------------
           Reset old results
        -------------------------------------------------------------- */

        if (resultsCard) {
            resultsCard.classList.add('hidden');
        }

        if (emptyState) {
            emptyState.classList.remove('hidden');
        }


        drawDummyWaveform();


        console.log(
            'Audio selected:',
            file.name,
            file.type,
            file.size
        );
    }


    /* ======================================================================
       9. MICROPHONE RECORDING
       ====================================================================== */

    function initializeRecording() {

        if (!startRecordBtn) {
            return;
        }

        startRecordBtn.addEventListener(
            'click',
            startRecording
        );

        if (stopRecordBtn) {

            stopRecordBtn.addEventListener(
                'click',
                stopRecording
            );
        }
    }


    async function startRecording() {

        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia
        ) {

            alert(
                'Microphone recording is not supported by this browser.'
            );

            return;
        }


        try {

            const stream =
                await navigator.mediaDevices.getUserMedia(
                    {
                        audio: {
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    }
                );


            mediaStream = stream;


            audioCtx =
                new (
                    window.AudioContext ||
                    window.webkitAudioContext
                )();


            recordingSampleRate =
                audioCtx.sampleRate;


            const source =
                audioCtx.createMediaStreamSource(
                    stream
                );


            scriptProcessor =
                audioCtx.createScriptProcessor(
                    4096,
                    1,
                    1
                );


            pcmBuffers = [];

            isWavRecording = true;


            scriptProcessor.onaudioprocess =
                event => {

                    if (!isWavRecording) {
                        return;
                    }

                    const input =
                        event.inputBuffer
                            .getChannelData(0);

                    pcmBuffers.push(
                        new Float32Array(input)
                    );
                };


            source.connect(
                scriptProcessor
            );

            scriptProcessor.connect(
                audioCtx.destination
            );


            recordSeconds = 0;

            if (recordTimer) {
                recordTimer.textContent =
                    '00:00';
            }


            if (recordIndicator) {
                recordIndicator.classList.add(
                    'recording'
                );
            }


            if (recordStatusText) {

                recordStatusText.textContent =
                    'Recording live audio... speak into microphone';
            }


            startRecordBtn.disabled = true;


            if (stopRecordBtn) {
                stopRecordBtn.disabled = false;
            }


            recordTimerInterval =
                setInterval(
                    () => {

                        recordSeconds++;


                        const minutes =
                            String(
                                Math.floor(
                                    recordSeconds / 60
                                )
                            ).padStart(2, '0');


                        const seconds =
                            String(
                                recordSeconds % 60
                            ).padStart(2, '0');


                        if (recordTimer) {

                            recordTimer.textContent =
                                `${minutes}:${seconds}`;
                        }


                        if (recordSeconds >= 60) {

                            stopRecording();
                        }

                    },
                    1000
                );

        } catch (error) {

            console.error(
                'Microphone error:',
                error
            );

            alert(
                `Microphone access error:\n\n${error.message}`
            );
        }
    }


    /* ======================================================================
       10. STOP RECORDING
       ====================================================================== */

    function stopRecording() {

        isWavRecording = false;

        clearInterval(
            recordTimerInterval
        );

        recordTimerInterval = null;


        if (scriptProcessor) {

            try {
                scriptProcessor.disconnect();
            } catch (error) {
                console.warn(
                    'ScriptProcessor cleanup failed:',
                    error
                );
            }

            scriptProcessor = null;
        }


        if (
            audioCtx &&
            audioCtx.state !== 'closed'
        ) {

            try {
                audioCtx.close();
            } catch (error) {
                console.warn(
                    'AudioContext cleanup failed:',
                    error
                );
            }
        }

        audioCtx = null;


        if (mediaStream) {

            mediaStream
                .getTracks()
                .forEach(track => {
                    track.stop();
                });

            mediaStream = null;
        }


        if (recordIndicator) {
            recordIndicator.classList.remove(
                'recording'
            );
        }


        if (startRecordBtn) {
            startRecordBtn.disabled = false;
        }


        if (stopRecordBtn) {
            stopRecordBtn.disabled = true;
        }


        if (recordStatusText) {

            recordStatusText.textContent =
                'Recording captured successfully!';
        }


        if (pcmBuffers.length === 0) {

            alert(
                'No audio data was captured. Please check your microphone.'
            );

            return;
        }


        /* --------------------------------------------------------------
           Encode PCM → WAV
        -------------------------------------------------------------- */

        const wavBlob =
            encodeWAV(
                pcmBuffers,
                recordingSampleRate
            );


        currentAudioBlob = wavBlob;


        currentAudioFile =
            new File(
                [wavBlob],
                `recorded_voice_${Date.now()}.wav`,
                {
                    type: 'audio/wav'
                }
            );


        /* --------------------------------------------------------------
           Preview
        -------------------------------------------------------------- */

        if (audioPlayer) {

            if (audioPlayer.src) {

                URL.revokeObjectURL(
                    audioPlayer.src
                );
            }

            const audioUrl =
                URL.createObjectURL(
                    wavBlob
                );

            audioPlayer.src =
                audioUrl;
        }


        if (fileNameDisplay) {

            fileNameDisplay.innerHTML =
                '<i class="fa-solid fa-microphone"></i> ' +
                'Recorded_Voice.wav';
        }


        if (audioPreviewCard) {

            audioPreviewCard.classList.remove(
                'hidden'
            );
        }


        if (resultsCard) {
            resultsCard.classList.add('hidden');
        }


        if (emptyState) {
            emptyState.classList.remove('hidden');
        }


        drawDummyWaveform();
    }


    /* ======================================================================
       11. WAV ENCODER
       ====================================================================== */

    function encodeWAV(
        samplesBlocks,
        sampleRate
    ) {

        let totalLength = 0;


        for (const block of samplesBlocks) {

            totalLength +=
                block.length;
        }


        const buffer =
            new ArrayBuffer(
                44 +
                totalLength * 2
            );


        const view =
            new DataView(buffer);


        function writeString(
            view,
            offset,
            string
        ) {

            for (
                let i = 0;
                i < string.length;
                i++
            ) {

                view.setUint8(
                    offset + i,
                    string.charCodeAt(i)
                );
            }
        }


        // RIFF
        writeString(
            view,
            0,
            'RIFF'
        );


        view.setUint32(
            4,
            36 + totalLength * 2,
            true
        );


        // WAVE
        writeString(
            view,
            8,
            'WAVE'
        );


        // fmt
        writeString(
            view,
            12,
            'fmt '
        );


        view.setUint32(
            16,
            16,
            true
        );


        // PCM
        view.setUint16(
            20,
            1,
            true
        );


        // Mono
        view.setUint16(
            22,
            1,
            true
        );


        // Sample rate
        view.setUint32(
            24,
            sampleRate,
            true
        );


        // Byte rate
        view.setUint32(
            28,
            sampleRate * 2,
            true
        );


        // Block align
        view.setUint16(
            32,
            2,
            true
        );


        // Bits per sample
        view.setUint16(
            34,
            16,
            true
        );


        // data
        writeString(
            view,
            36,
            'data'
        );


        view.setUint32(
            40,
            totalLength * 2,
            true
        );


        // PCM samples
        let offset = 44;


        for (const block of samplesBlocks) {

            for (
                let i = 0;
                i < block.length;
                i++
            ) {

                let sample =
                    Math.max(
                        -1,
                        Math.min(
                            1,
                            block[i]
                        )
                    );


                const value =
                    sample < 0
                        ? sample * 0x8000
                        : sample * 0x7FFF;


                view.setInt16(
                    offset,
                    value,
                    true
                );


                offset += 2;
            }
        }


        return new Blob(
            [buffer],
            {
                type: 'audio/wav'
            }
        );
    }


    /* ======================================================================
       12. CLEAR AUDIO
       ====================================================================== */

    function initializeClearButton() {

        if (!clearAudioBtn) {
            return;
        }


        clearAudioBtn.addEventListener(
            'click',
            () => {

                currentAudioFile = null;
                currentAudioBlob = null;


                if (audioPlayer) {

                    if (audioPlayer.src) {

                        try {
                            URL.revokeObjectURL(
                                audioPlayer.src
                            );
                        } catch (error) {}
                    }

                    audioPlayer.src = '';
                }


                if (audioFileInput) {
                    audioFileInput.value = '';
                }


                if (audioPreviewCard) {
                    audioPreviewCard.classList.add(
                        'hidden'
                    );
                }


                if (resultsCard) {
                    resultsCard.classList.add(
                        'hidden'
                    );
                }


                if (emptyState) {
                    emptyState.classList.remove(
                        'hidden'
                    );
                }


                console.log(
                    'Audio cleared.'
                );
            }
        );
    }


    /* ======================================================================
       13. AUDIO ANALYSIS
       ====================================================================== */

    function initializeAnalysis() {

        if (!analyzeBtn) {
            return;
        }


        analyzeBtn.addEventListener(
            'click',
            analyzeAudio
        );
    }


    async function analyzeAudio() {

        if (isAnalyzing) {
            return;
        }


        if (!currentAudioBlob) {

            alert(
                'Please select or record an audio file first.'
            );

            return;
        }


        isAnalyzing = true;


        /* --------------------------------------------------------------
           Disable analyze button
        -------------------------------------------------------------- */

        analyzeBtn.disabled = true;


        const originalButtonHTML =
            analyzeBtn.innerHTML;


        analyzeBtn.innerHTML =
            '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';


        /* --------------------------------------------------------------
           Show loading
        -------------------------------------------------------------- */

        if (audioPreviewCard) {
            audioPreviewCard.classList.add(
                'hidden'
            );
        }


        if (loadingOverlay) {
            loadingOverlay.classList.remove(
                'hidden'
            );
        }


        animateProgressBar();


        /* --------------------------------------------------------------
           Prepare FormData
        -------------------------------------------------------------- */

        const formData =
            new FormData();


        /*
         * IMPORTANT
         * Backend expects:
         *
         * request.files["audio"]
         *
         * Therefore the field MUST be "audio".
         */

        const uploadFile =
            currentAudioFile ||
            currentAudioBlob;


        const filename =
            currentAudioFile?.name ||
            'recorded_voice.wav';


        formData.append(
            'audio',
            uploadFile,
            filename
        );


        console.log(
            'Uploading audio:',
            filename
        );


        try {

            /* ----------------------------------------------------------
               Send request
            ---------------------------------------------------------- */

            const response =
                await fetchWithTimeout(
                    `${API_BASE_URL}/analyze`,
                    {
                        method: 'POST',
                        body: formData,
                        headers: {
                            'Accept':
                                'application/json'
                        }
                    },
                    120000
                );


            /* ----------------------------------------------------------
               Read response safely
            ---------------------------------------------------------- */

            const contentType =
                response.headers.get(
                    'content-type'
                ) || '';


            let data;


            if (
                contentType.includes(
                    'application/json'
                )
            ) {

                data =
                    await response.json();

            } else {

                const text =
                    await response.text();


                console.error(
                    'Non-JSON server response:',
                    text
                );


                throw new Error(
                    `Server returned a non-JSON response ` +
                    `(HTTP ${response.status}). ` +
                    `Response: ${text.substring(0, 500)}`
                );
            }


            console.log(
                'Analysis API response:',
                data
            );


            /* ----------------------------------------------------------
               Hide loading
            ---------------------------------------------------------- */

            if (loadingOverlay) {
                loadingOverlay.classList.add(
                    'hidden'
                );
            }


            if (audioPreviewCard) {
                audioPreviewCard.classList.remove(
                    'hidden'
                );
            }


            /* ----------------------------------------------------------
               Handle backend error
            ---------------------------------------------------------- */

            if (
                !response.ok ||
                !data ||
                data.success === false
            ) {

                const message =
                    data?.error ||
                    data?.message ||
                    `Analysis failed with HTTP ${response.status}`;


                throw new Error(message);
            }


            /* ----------------------------------------------------------
               Validate successful response
            ---------------------------------------------------------- */

            if (
                typeof data.success !== 'undefined' &&
                data.success !== true
            ) {

                throw new Error(
                    'The server did not return a successful analysis result.'
                );
            }


            /* ----------------------------------------------------------
               Render results
            ---------------------------------------------------------- */

            renderResults(data);


        } catch (error) {

            console.error(
                'Voice analysis failed:',
                error
            );


            if (loadingOverlay) {
                loadingOverlay.classList.add(
                    'hidden'
                );
            }


            if (audioPreviewCard) {
                audioPreviewCard.classList.remove(
                    'hidden'
                );
            }


            showAnalysisError(
                error
            );


        } finally {

            isAnalyzing = false;

            analyzeBtn.disabled = false;

            analyzeBtn.innerHTML =
                originalButtonHTML;
        }
    }


    /* ======================================================================
       14. FETCH WITH TIMEOUT
       ====================================================================== */

    async function fetchWithTimeout(
        url,
        options = {},
        timeoutMs = 120000
    ) {

        const controller =
            new AbortController();


        const timeoutId =
            setTimeout(
                () => {
                    controller.abort();
                },
                timeoutMs
            );


        try {

            const response =
                await fetch(
                    url,
                    {
                        ...options,
                        signal:
                            controller.signal
                    }
                );


            return response;

        } catch (error) {

            if (
                error.name ===
                'AbortError'
            ) {

                throw new Error(
                    'Voice analysis timed out after 120 seconds. ' +
                    'The server may be processing a large audio file.'
                );
            }


            throw error;

        } finally {

            clearTimeout(
                timeoutId
            );
        }
    }


    /* ======================================================================
       15. ANALYSIS ERROR DISPLAY
       ====================================================================== */

    function showAnalysisError(error) {

        let message =
            error?.message ||
            'An unexpected error occurred during voice analysis.';


        if (
            message.includes(
                'Failed to fetch'
            )
        ) {

            message =
                'Could not connect to the Voice Authenticity API. ' +
                'Please check your internet connection and try again.';
        }


        if (
            message.includes(
                'NetworkError'
            )
        ) {

            message =
                'Network error while contacting the analysis server. ' +
                'Please try again.';
        }


        alert(
            `Analysis Error:\n\n${message}`
        );
    }


    /* ======================================================================
       16. PROGRESS BAR
       ====================================================================== */

    function animateProgressBar() {

        if (!progressFill) {
            return;
        }


        progressFill.style.width =
            '0%';


        let width = 0;


        const interval =
            setInterval(
                () => {

                    if (
                        !loadingOverlay ||
                        loadingOverlay.classList.contains(
                            'hidden'
                        )
                    ) {

                        clearInterval(
                            interval
                        );

                        return;
                    }


                    if (width >= 90) {

                        clearInterval(
                            interval
                        );

                        return;
                    }


                    width += 5;


                    progressFill.style.width =
                        `${width}%`;

                },
                500
            );
    }


    /* ======================================================================
       17. RENDER RESULTS
       ====================================================================== */

    function renderResults(data) {

        console.log(
            'Rendering analysis results:',
            data
        );


        /* --------------------------------------------------------------
           Basic validation
        -------------------------------------------------------------- */

        if (!data) {

            throw new Error(
                'Empty analysis response received from server.'
            );
        }


        if (emptyState) {
            emptyState.classList.add(
                'hidden'
            );
        }


        if (resultsCard) {
            resultsCard.classList.remove(
                'hidden'
            );
        }


        /* --------------------------------------------------------------
           Human / AI verdict
        -------------------------------------------------------------- */

        const isHuman =
            Boolean(data.is_human);


        if (verdictBanner) {

            verdictBanner.className =
                isHuman
                    ? 'verdict-banner verdict-human'
                    : 'verdict-banner verdict-ai';
        }


        if (verdictIconI) {

            verdictIconI.className =
                isHuman
                    ? 'fa-solid fa-user-check'
                    : 'fa-solid fa-robot';
        }


        if (predictionTitle) {

            predictionTitle.textContent =
                isHuman
                    ? 'LIKELY HUMAN VOICE'
                    : 'AI-GENERATED / MANIPULATED VOICE';


            predictionTitle.style.color =
                isHuman
                    ? 'var(--color-human)'
                    : 'var(--color-ai)';
        }


        /* --------------------------------------------------------------
           Confidence
        -------------------------------------------------------------- */

        if (confidenceVal) {

            const confidence =
                Number(
                    data.confidence
                );


            confidenceVal.textContent =
                Number.isFinite(confidence)
                    ? `${round(confidence, 1)}%`
                    : 'N/A';
        }


        /* --------------------------------------------------------------
           Model status
        -------------------------------------------------------------- */

        if (data.demo_mode) {

            if (calibratedStatusTag) {

                calibratedStatusTag.textContent =
                    'Uncalibrated / Demo';
            }


            if (demoNoticeBox) {

                demoNoticeBox.classList.remove(
                    'hidden'
                );
            }


            if (demoNoticeText) {

                demoNoticeText.textContent =
                    data.status_note ||
                    'The system is operating in demo mode.';
            }

        } else {

            if (calibratedStatusTag) {

                calibratedStatusTag.textContent =
                    'Trained ML Estimate';
            }


            if (demoNoticeBox) {

                demoNoticeBox.classList.add(
                    'hidden'
                );
            }
        }


        /* --------------------------------------------------------------
           Audio metrics
        -------------------------------------------------------------- */

        const metrics =
            data.audio_metrics || {};


        if (mDuration) {

            mDuration.textContent =
                formatNumber(
                    metrics.duration_seconds,
                    2,
                    's'
                );
        }


        if (mSr) {

            mSr.textContent =
                formatNumber(
                    metrics.sample_rate_hz,
                    0,
                    'Hz'
                );
        }


        if (mTime) {

            mTime.textContent =
                formatNumber(
                    data.processing_time_ms,
                    0,
                    'ms'
                );
        }


        if (mModel) {

            mModel.textContent =
                data.model_used ||
                'Unknown';
        }


        /* --------------------------------------------------------------
           Feature summary
        -------------------------------------------------------------- */

        const features =
            data.feature_summary || {};


        if (fPitch) {

            fPitch.textContent =
                formatNumber(
                    features.pitch_f0_mean_hz,
                    2,
                    'Hz'
                );
        }


        if (fPitchVar) {

            fPitchVar.textContent =
                safeValue(
                    features.pitch_variation_index
                );
        }


        if (fCentroid) {

            fCentroid.textContent =
                formatNumber(
                    features.spectral_centroid_hz,
                    2,
                    'Hz'
                );
        }


        if (fRolloff) {

            fRolloff.textContent =
                formatNumber(
                    features.spectral_rolloff_hz,
                    2,
                    'Hz'
                );
        }


        if (fZcr) {

            fZcr.textContent =
                safeValue(
                    features.zero_crossing_rate
                );
        }


        if (fMfccVar) {

            fMfccVar.textContent =
                safeValue(
                    features.mfcc_variance
                );
        }


        /* --------------------------------------------------------------
           Waveform
        -------------------------------------------------------------- */

        if (
            Array.isArray(
                data.waveform_data
            ) &&
            data.waveform_data.length > 0
        ) {

            drawCustomWaveform(
                data.waveform_data
            );
        }


        /* --------------------------------------------------------------
           XAI
        -------------------------------------------------------------- */

        renderXAI(
            data.xai_explanation
        );


        /* --------------------------------------------------------------
           Scroll
        -------------------------------------------------------------- */

        if (resultsCard) {

            resultsCard.scrollIntoView(
                {
                    behavior: 'smooth',
                    block: 'start'
                }
            );
        }
    }


    /* ======================================================================
       18. XAI RENDERING
       ====================================================================== */

    function renderXAI(xai) {

        if (!xai) {

            if (xaiSummaryText) {

                xaiSummaryText.textContent =
                    'No explainability information was returned by the analysis service.';
            }

            if (xaiFactorsContainer) {

                xaiFactorsContainer.innerHTML =
                    '';
            }

            return;
        }


        if (xaiSummaryText) {

            xaiSummaryText.textContent =
                xai.summary ||
                'The analysis system evaluated acoustic characteristics of the recording.';
        }


        if (!xaiFactorsContainer) {
            return;
        }


        xaiFactorsContainer.innerHTML =
            '';


        const factors =
            Array.isArray(xai.factors)
                ? xai.factors
                : [];


        factors.forEach(
            factor => {

                const card =
                    document.createElement(
                        'div'
                    );


                card.className =
                    'xai-factor-card';


                const title =
                    document.createElement(
                        'div'
                    );

                title.className =
                    'xai-title';

                title.textContent =
                    factor.feature ||
                    'Acoustic Feature';


                const observation =
                    document.createElement(
                        'div'
                    );

                observation.className =
                    'xai-obs';

                observation.textContent =
                    factor.observation ||
                    'No observation available.';


                const impact =
                    document.createElement(
                        'div'
                    );

                impact.className =
                    'xai-impact';

                impact.textContent =
                    factor.impact ||
                    'Feature contribution included in the analysis.';


                card.appendChild(title);
                card.appendChild(observation);
                card.appendChild(impact);


                xaiFactorsContainer.appendChild(
                    card
                );
            }
        );
    }


    /* ======================================================================
       19. WAVEFORM
       ====================================================================== */

    function drawDummyWaveform() {

        const points = [];


        for (
            let i = 0;
            i < 80;
            i++
        ) {

            points.push(
                Math.random() * 0.7 + 0.1
            );
        }


        drawCustomWaveform(
            points
        );
    }


    function drawCustomWaveform(points) {

        if (!waveformCanvas) {
            return;
        }


        if (
            !Array.isArray(points) ||
            points.length === 0
        ) {

            return;
        }


        const ctx =
            waveformCanvas.getContext(
                '2d'
            );


        if (!ctx) {
            return;
        }


        const width =
            waveformCanvas.offsetWidth ||
            600;


        const height = 80;


        waveformCanvas.width =
            width;


        waveformCanvas.height =
            height;


        ctx.clearRect(
            0,
            0,
            width,
            height
        );


        const barWidth =
            width / points.length;


        const center =
            height / 2;


        ctx.fillStyle =
            '#38bdf8';


        points.forEach(
            (value, index) => {

                const normalized =
                    Math.max(
                        0,
                        Math.min(
                            1,
                            Number(value) || 0
                        )
                    );


                const barHeight =
                    normalized *
                    (height * 0.8);


                const x =
                    index *
                    barWidth;


                const y =
                    center -
                    barHeight / 2;


                ctx.beginPath();


                if (
                    typeof ctx.roundRect ===
                    'function'
                ) {

                    ctx.roundRect(
                        x + 1,
                        y,
                        Math.max(
                            1,
                            barWidth - 2
                        ),
                        Math.max(
                            1,
                            barHeight
                        ),
                        2
                    );

                } else {

                    ctx.rect(
                        x + 1,
                        y,
                        Math.max(
                            1,
                            barWidth - 2
                        ),
                        Math.max(
                            1,
                            barHeight
                        )
                    );
                }


                ctx.fill();
            }
        );
    }


    /* ======================================================================
       20. HELPER FUNCTIONS
       ====================================================================== */

    function round(
        number,
        decimals = 2
    ) {

        const value =
            Number(number);


        if (!Number.isFinite(value)) {
            return 0;
        }


        const multiplier =
            Math.pow(
                10,
                decimals
            );


        return Math.round(
            value * multiplier
        ) / multiplier;
    }


    function safeValue(value) {

        if (
            value === null ||
            value === undefined ||
            value === ''
        ) {

            return 'N/A';
        }


        if (
            typeof value === 'number' &&
            !Number.isFinite(value)
        ) {

            return 'N/A';
        }


        return value;
    }


    function formatNumber(
        value,
        decimals = 2,
        suffix = ''
    ) {

        const number =
            Number(value);


        if (!Number.isFinite(number)) {

            return 'N/A';
        }


        return `${round(
            number,
            decimals
        )} ${suffix}`.trim();
    }


    function escapeHtml(value) {

        const div =
            document.createElement(
                'div'
            );


        div.textContent =
            value ?? '';


        return div.innerHTML;
    }


    /* ======================================================================
       21. WINDOW RESIZE
       ====================================================================== */

    window.addEventListener(
        'resize',
        () => {

            if (
                waveformCanvas &&
                resultsCard &&
                !resultsCard.classList.contains(
                    'hidden'
                )
            ) {

                // Redraw only if backend waveform exists
                // Otherwise leave current visualization.
            }
        }
    );


    /* ======================================================================
       22. PAGE CLEANUP
       ====================================================================== */

    window.addEventListener(
        'beforeunload',
        () => {

            clearInterval(
                recordTimerInterval
            );


            if (mediaStream) {

                mediaStream
                    .getTracks()
                    .forEach(track => {
                        track.stop();
                    });
            }


            if (
                audioCtx &&
                audioCtx.state !== 'closed'
            ) {

                try {
                    audioCtx.close();
                } catch (error) {}
            }
        }
    );

});