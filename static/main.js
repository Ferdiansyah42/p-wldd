/* ═══════════════════════════════════════════════════════
   BPS-Style Academic Dashboard — Frontend Logic (Vanilla JS)
   ═══════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", function() {
    initDatasetUpload();
    initPreprocessing();
    initTrainingSSE();
    initRecipientSearch();
});

// ──────────────────────────────────────────────
// 1. DATASET UPLOAD & SAMPLE LOADING
// ──────────────────────────────────────────────
function initDatasetUpload() {
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const btnBrowse = document.getElementById("btn-browse-file");
    const btnLoadSample = document.getElementById("btn-load-sample");
    const statusAlert = document.getElementById("upload-status");

    if (!uploadZone) return;

    // Helper to show upload alerts
    function showAlert(message, isSuccess = true) {
        statusAlert.style.display = "block";
        statusAlert.style.backgroundColor = isSuccess ? "#e6f4ea" : "#fce8e6";
        statusAlert.style.color = isSuccess ? "#137333" : "#c5221f";
        statusAlert.style.border = `1px solid ${isSuccess ? "#13733350" : "#c5221f50"}`;
        statusAlert.innerHTML = message;
    }

    // Trigger browse
    btnBrowse.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

    // Drag and drop handlers
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "var(--secondary-color)";
        uploadZone.style.backgroundColor = "rgba(59, 130, 246, 0.05)";
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.style.borderColor = "var(--border-color)";
        uploadZone.style.backgroundColor = "#fcfdfe";
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "var(--border-color)";
        uploadZone.style.backgroundColor = "#fcfdfe";
        handleFiles(e.dataTransfer.files);
    });

    // Upload function
    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        const formData = new FormData();
        formData.append("file", file);

        showAlert("Sedang mengunggah berkas CSV...", true);

        fetch("/api/upload", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showAlert(`✅ Sukses: ${data.message} Halaman akan dimuat ulang.`, true);
                setTimeout(() => window.location.reload(), 1500);
            } else {
                showAlert(`❌ Error: ${data.message}`, false);
            }
        })
        .catch(err => {
            showAlert(`❌ Gagal mengunggah: ${err}`, false);
        });
    }

    // Load sample dataset
    btnLoadSample.addEventListener("click", () => {
        showAlert("Sedang mengunduh dan menghasilkan dataset sampel...", true);
        btnLoadSample.disabled = true;

        fetch("/api/load-sample", {
            method: "POST"
        })
        .then(res => res.json())
        .then(data => {
            btnLoadSample.disabled = false;
            if (data.success) {
                showAlert(`✅ Sukses: ${data.message} Halaman akan dimuat ulang.`, true);
                setTimeout(() => window.location.reload(), 1200);
            } else {
                showAlert(`❌ Error: ${data.message}`, false);
            }
        })
        .catch(err => {
            btnLoadSample.disabled = false;
            showAlert(`❌ Gagal memuat data sampel: ${err}`, false);
        });
    });
}

// ──────────────────────────────────────────────
// 2. PREPROCESSING PIPELINE
// ──────────────────────────────────────────────
function initPreprocessing() {
    const btnRunPreprocess = document.getElementById("btn-run-preprocess");
    const summaryCard = document.getElementById("preprocess-summary-card");
    const valDup = document.getElementById("val-dup");
    const valFeatures = document.getElementById("val-features");

    if (!btnRunPreprocess) return;

    btnRunPreprocess.addEventListener("click", () => {
        btnRunPreprocess.disabled = true;
        btnRunPreprocess.innerHTML = "⚡ Memproses Data...";

        fetch("/api/run-preprocess", {
            method: "POST"
        })
        .then(res => res.json())
        .then(data => {
            btnRunPreprocess.disabled = false;
            btnRunPreprocess.innerHTML = "⚡ Jalankan Preprocessing Data";

            if (data.success) {
                // Set all checklist items as checked
                document.querySelectorAll(".checklist-item").forEach(item => {
                    item.classList.add("checked");
                });
                
                // Update badges
                document.getElementById("badge-clean").innerHTML = "Dibersihkan";
                document.getElementById("badge-clean").className = "badge badge-success";
                
                document.getElementById("badge-norm").innerHTML = "Normal";
                document.getElementById("badge-norm").className = "badge badge-success";
                
                document.getElementById("badge-split").innerHTML = "Terbagi (80:20)";
                document.getElementById("badge-split").className = "badge badge-success";

                // Show details card
                summaryCard.style.display = "block";
                valDup.innerHTML = data.n_dup;
                valFeatures.innerHTML = data.n_features;
                
                alert("✅ Preprocessing selesai dilakukan!");
            } else {
                alert(`❌ Gagal preprocessing: ${data.message}`);
            }
        })
        .catch(err => {
            btnRunPreprocess.disabled = false;
            btnRunPreprocess.innerHTML = "⚡ Jalankan Preprocessing Data";
            alert(`❌ Error koneksi: ${err}`);
        });
    });
}

// ──────────────────────────────────────────────
// 3. THREADED TRAINING OVER SERVER-SENT EVENTS
// ──────────────────────────────────────────────
function initTrainingSSE() {
    const btnRunGA = document.getElementById("btn-run-ga");
    const btnRunSVM = document.getElementById("btn-run-svm");
    const trainConsole = document.getElementById("train-console");
    const statusTrainingLabel = document.getElementById("svm-training-status");
    
    // Live GA panels
    const gaLivePanel = document.getElementById("ga-live-status");
    const liveGen = document.getElementById("live-gen");
    const liveMaxGen = document.getElementById("live-max-gen");
    const liveProgressFill = document.getElementById("live-progress-fill");
    const liveBestFit = document.getElementById("live-best-fit");
    const liveAvgFit = document.getElementById("live-avg-fit");
    const liveNFeat = document.getElementById("live-n-feat");
    const liveCache = document.getElementById("live-cache");
    const liveEta = document.getElementById("live-eta");

    // Only run if training components exist on page
    if (!btnRunGA && !btnRunSVM) return;

    const isGAPage = btnRunGA !== null;
    const triggerBtn = isGAPage ? btnRunGA : btnRunSVM;

    triggerBtn.addEventListener("click", () => {
        // Disable buttons
        triggerBtn.disabled = true;
        
        // SVM and GA params parsing
        let svmKernel = "rbf";
        let svmC = 1.0;
        let svmGamma = "scale";
        let gaPop = 30;
        let gaGen = 50;
        let gaCrossover = 0.8;
        let gaMutation = 0.1;
        let gaTournament = 3;
        let gaCV = 3;
        let gaElitism = true;
        let gaEarlyStop = 10;

        // Fetch inputs if they exist (depending on which page we are on)
        const elKernel = document.getElementById("svm-kernel");
        if (elKernel) svmKernel = elKernel.value;
        
        const elC = document.getElementById("svm-c");
        if (elC) svmC = elC.value;
        
        const elGamma = document.getElementById("svm-gamma");
        if (elGamma) svmGamma = elGamma.value;
        
        const elPop = document.getElementById("ga-pop-size");
        if (elPop) gaPop = elPop.value;
        
        const elGen = document.getElementById("ga-generations");
        if (elGen) gaGen = elGen.value;
        
        const elCross = document.getElementById("ga-crossover");
        if (elCross) gaCrossover = elCross.value;
        
        const elMut = document.getElementById("ga-mutation");
        if (elMut) gaMutation = elMut.value;
        
        const elTour = document.getElementById("ga-tournament");
        if (elTour) gaTournament = elTour.value;
        
        const elCV = document.getElementById("ga-cv-folds");
        if (elCV) gaCV = elCV.value;
        
        const elElit = document.getElementById("ga-elitism");
        if (elElit) gaElitism = elElit.checked;

        const elEarly = document.getElementById("ga-early-stop");
        if (elEarly) gaEarlyStop = elEarly.value;

        // Clear console log
        if (trainConsole) {
            trainConsole.innerHTML = '<div class="console-line success">&gt; Memulai inisialisasi koneksi EventSource...</div>';
        }
        
        // Show status indicators
        if (statusTrainingLabel) statusTrainingLabel.style.display = "flex";
        if (gaLivePanel) gaLivePanel.style.display = "block";

        // Query params
        const queryParams = new URLSearchParams({
            kernel: svmKernel,
            C: svmC,
            gamma: svmGamma,
            pop_size: gaPop,
            generations: gaGen,
            crossover_rate: gaCrossover,
            mutation_rate: gaMutation,
            tournament_k: gaTournament,
            cv_folds: gaCV,
            elitism: gaElitism,
            early_stop: gaEarlyStop
        });

        // Initialize EventSource SSE
        const eventSource = new EventSource(`/api/train-stream?${queryParams.toString()}`);

        eventSource.onmessage = function(event) {
            const msg = JSON.parse(event.data);
            
            if (msg.type === "ping") return;

            if (msg.type === "log") {
                appendConsoleLine(msg.message);
            } 
            else if (msg.type === "progress") {
                const data = msg.data;
                
                // Update live GA panel if visible
                if (gaLivePanel) {
                    liveGen.innerHTML = data.generation;
                    liveMaxGen.innerHTML = data.max_generations;
                    liveBestFit.innerHTML = data.best_fitness.toFixed(4);
                    liveAvgFit.innerHTML = data.avg_fitness.toFixed(4);
                    liveNFeat.innerHTML = `${data.n_features}/${data.total_features}`;
                    liveCache.innerHTML = `${data.cache_hits}/${data.pop_size}`;
                    liveEta.innerHTML = Math.round(data.eta);

                    const pct = (data.generation / data.max_generations) * 100;
                    liveProgressFill.style.width = `${pct}%`;
                }

                // Add log representation
                appendConsoleLine(`Generasi ${data.generation}/${data.max_generations} - Best Fitness: ${data.best_fitness.toFixed(4)} - Fitur: ${data.n_features}`);
            }
            else if (msg.type === "done") {
                appendConsoleLine(`✅ SUKSES: ${msg.message}`, "success");
                eventSource.close();
                
                if (statusTrainingLabel) statusTrainingLabel.style.display = "none";
                triggerBtn.disabled = false;
                
                // Dynamically update features table in ga_selection page
                updateFeaturesTableUI(msg.state);
                
                alert("🎉 Pelatihan Model & Seleksi Fitur Berhasil Selesai!");
            }
            else if (msg.type === "error") {
                appendConsoleLine(`❌ ERROR: ${msg.message}`, "error");
                eventSource.close();
                
                if (statusTrainingLabel) statusTrainingLabel.style.display = "none";
                triggerBtn.disabled = false;
                
                alert(`❌ Terjadi Kesalahan: ${msg.message}`);
            }
        };

        eventSource.onerror = function() {
            appendConsoleLine(`❌ Error koneksi EventSource. Saluran ditutup.`, "error");
            eventSource.close();
            if (statusTrainingLabel) statusTrainingLabel.style.display = "none";
            triggerBtn.disabled = false;
        };
    });

    // Helper functions for console
    function appendConsoleLine(text, className = "") {
        if (!trainConsole) return;
        const line = document.createElement("div");
        line.className = `console-line ${className}`;
        line.textContent = `> ${text}`;
        trainConsole.appendChild(line);
        trainConsole.scrollTop = trainConsole.scrollHeight; // Scroll to bottom
    }

    // Dynamic UI Updates
    function updateFeaturesTableUI(state) {
        const table = document.getElementById("features-table");
        if (!table) return;

        const selectedSet = new Set(state.selected_features);
        
        document.querySelectorAll("#features-table tbody tr.feature-row").forEach(row => {
            const fName = row.getAttribute("data-name");
            const isSel = selectedSet.has(fName);
            const statusCell = row.querySelector(".feature-status-cell");
            
            if (statusCell) {
                statusCell.innerHTML = isSel 
                    ? '<span class="badge badge-success">✓ DIPILIH</span>'
                    : '<span class="badge badge-danger">✗ DIELIMINASI</span>';
            }
        });
    }
}

// ──────────────────────────────────────────────
// 4. RECIPIENT SEARCH (NIK / NAMA)
// ──────────────────────────────────────────────
function initRecipientSearch() {
    const btnSearch = document.getElementById("btn-search-recipient");
    const queryInput = document.getElementById("search-query");
    const resultsContainer = document.getElementById("search-results-container");
    const emptyState = document.getElementById("search-empty-state");
    const errorState = document.getElementById("search-error-state");
    const errorTitle = document.getElementById("search-error-title");
    const errorDesc = document.getElementById("search-error-desc");
    const modelStatusAlert = document.getElementById("search-model-status");
    const resultTemplate = document.getElementById("result-card-template");

    if (!btnSearch) return;

    btnSearch.addEventListener("click", () => {
        const query = queryInput.value.trim();
        if (!query) {
            alert("Masukkan NIK atau Nama terlebih dahulu!");
            return;
        }

        btnSearch.disabled = true;
        btnSearch.innerHTML = "🔍 Mencari...";
        
        // Hide states
        resultsContainer.style.display = "none";
        emptyState.style.display = "none";
        errorState.style.display = "none";
        modelStatusAlert.style.display = "none";
        
        fetch(`/api/search?query=${encodeURIComponent(query)}`)
        .then(res => res.json())
        .then(data => {
            btnSearch.disabled = false;
            btnSearch.innerHTML = "🔍 Cari Penduduk";

            if (data.success) {
                // Show model trained alert if needed
                if (!data.model_active) {
                    modelStatusAlert.style.display = "block";
                }

                // Clear previous results
                resultsContainer.innerHTML = "";
                resultsContainer.style.display = "block";

                // Loop results and clone template
                data.results.forEach(res => {
                    const clone = resultTemplate.content.cloneNode(true);
                    
                    clone.querySelector(".res-nama").textContent = res.nama;
                    clone.querySelector(".res-nik").textContent = res.nik;
                    
                    const badge = clone.querySelector(".res-status-badge");
                    badge.textContent = res.pred_label;
                    
                    const isLayak = res.pred_label.toLowerCase() === "layak" || res.pred_label === "1";
                    badge.className = `badge ${isLayak ? 'badge-success' : 'badge-danger'}`;
                    
                    clone.querySelector(".res-confidence").textContent = res.confidence;
                    
                    // Detail grid
                    const detailsGrid = clone.querySelector(".grid-3");
                    
                    // Helper to format values
                    function formatDetailVal(key, val) {
                        if (key.toLowerCase() === "pendapatan") {
                            try {
                                return `Rp ${parseInt(val).toLocaleString()}`;
                            } catch (e) {
                                return val;
                            }
                        }
                        if (key.toLowerCase() === "luas rumah") {
                            return `${val} m²`;
                        }
                        return val;
                    }
                    
                    for (const [key, val] of Object.entries(res.details)) {
                        const item = document.createElement("div");
                        item.style.padding = "10px";
                        item.style.border = "1px solid var(--border-color)";
                        item.style.borderRadius = "var(--border-radius-sm)";
                        item.style.backgroundColor = "var(--bg-color)";
                        
                        item.innerHTML = `
                            <span style="font-size: 0.725rem; display: block; color: var(--text-light); text-transform: uppercase; font-weight: 600;">${key}</span>
                            <span style="font-size: 0.9rem; font-weight: 600;">${formatDetailVal(key, val)}</span>
                        `;
                        detailsGrid.appendChild(item);
                    }

                    resultsContainer.appendChild(clone);
                });
            } else {
                errorState.style.display = "block";
                errorTitle.textContent = "Data tidak ditemukan";
                errorDesc.textContent = data.message;
            }
        })
        .catch(err => {
            btnSearch.disabled = false;
            btnSearch.innerHTML = "🔍 Cari Penduduk";
            errorState.style.display = "block";
            errorTitle.textContent = "Error Koneksi";
            errorDesc.textContent = `Terjadi kesalahan saat menghubungi server: ${err}`;
        });
    });
}
