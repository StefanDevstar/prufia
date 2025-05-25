
    let matchingResults = [];    
    const uploadForm = document.getElementById('uploadForm');
    let uploadedFilename='';
    function toggleDetails(button) {
        const item = button.closest('.history-item');
        item.classList.toggle('expanded');
        button.classList.toggle('rotate-180');
    }

    const socket = io({
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        query: { 'user-type': 'admin' }
    });

    const socketStatus = document.getElementById('socket-status');
    socket.on('connect', () => {
        console.log("I am joined")
        socket.emit('join-admin-room');
    });

    socket.on('disconnect', () => {
    });

    socket.on('connect_error', (error) => {
    });

    socket.on('progress', (data) => {
        const progressBar = document.getElementById('progress-bar');
        const statusNote = document.getElementById('progress-status');
        progressBar.style.width = `${data.value}%`;
        statusNote.textContent = `${data.value}% - ${data.func_name} was analyzed.`;

        if (data.value >= 99) {
            setTimeout(() => {
                progressBar.style.width = '0%';
                statusNote.textContent = 'Completed';
            }, 2000); 
        }  
    });
    const dropdownInput = document.querySelector('.dropdown-input');
    const dropdownList = document.querySelector('.dropdown-list');
    const checkboxes = document.querySelectorAll('.dropdown-item input[type="checkbox"]');
    const selectAll = document.getElementById('select-all');
    
    function initialize() {
        checkboxes.forEach(checkbox => {
            checkbox.checked = true; 
        });
        updateSelectedOptions();
    }
    
    dropdownInput.addEventListener('click', function() {
        dropdownList.classList.toggle('show');
    });
    
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.dropdown-container')) {
            dropdownList.classList.remove('show');
        }
    });
    
    function updateSelectedOptions() {
        const selectedOptions = [];
        checkboxes.forEach(checkbox => {
            if (checkbox !== selectAll && checkbox.checked) {
                selectedOptions.push(checkbox.value);
            }
        });
        
        dropdownInput.value = selectedOptions.join(', ');
        
        // Update "Select All" checkbox state
        const allChecked = [...checkboxes].slice(1).every(checkbox => checkbox.checked);
        selectAll.checked = allChecked;
        selectAll.indeterminate = !allChecked && selectedOptions.length > 0;
    }
    
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this === selectAll) {
                checkboxes.forEach(cb => {
                    if (cb !== selectAll) {
                        cb.checked = selectAll.checked;
                    }
                });
            }
            updateSelectedOptions();
        });
    });
    
    initialize();
    document.getElementById('clearSelectionBtn').addEventListener('click', function() {
        document.getElementById('assignmentFiles').value = '';
        const filePreviews = document.getElementById('filePreviews');
        filePreviews.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-file-alt"></i>
                <p>No files selected</p>
            </div>
        `;
        const originalText = this.innerHTML;
        this.innerHTML = '<i class="fas fa-check"></i> Cleared';
        setTimeout(() => {
            this.innerHTML = originalText;
        }, 1500);
    });
    document.getElementById('assignmentFiles').addEventListener('change', function(e) {
        const files = e.target.files;
        const previewContainer = document.getElementById('filePreviews');
        previewContainer.innerHTML = ''; 
        
        if (files.length === 0) return;
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const fileExt = file.name.split('.').pop().toLowerCase();
            const previewDiv = document.createElement('div');
            previewDiv.className = 'file-preview';
            previewDiv.dataset.fileIndex = i; 
            const deleteBtn = document.createElement('div');
            deleteBtn.className = 'delete-preview';
            deleteBtn.innerHTML = '×';
            deleteBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                removeFileFromList(i);
                previewDiv.remove();
            });
            
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewDiv.innerHTML = `
                        <div><i class="fas fa-file-image img-icon file-icon"></i></div>
                        <img src="${e.target.result}" class="preview-image" alt="${file.name}">
                        <div class="file-name">${truncateFileName(file.name)}</div>
                        <div class="file-size">${formatFileSize(file.size)}</div>
                    `;
                    previewDiv.appendChild(deleteBtn);
                };
                reader.readAsDataURL(file);
            } else {
                let iconClass = 'fa-file';
                let extraClass = '';
                if (fileExt === 'pdf') {
                    iconClass = 'fa-file-pdf';
                    extraClass = 'pdf-icon';
                } 
                else if (fileExt === 'doc' || fileExt === 'docx') {
                    iconClass = 'fa-file-word';
                    extraClass = 'doc-icon';
                }
                else if (fileExt === 'xls' || fileExt === 'xlsx') {
                    iconClass = 'fa-file-excel';
                    extraClass = 'xls-icon';
                }                    
                else if (fileExt === 'txt') iconClass = 'fa-file-alt';
                
                previewDiv.innerHTML = `
                    <i class="fas ${iconClass} file-icon ${extraClass}"></i>
                    <div class="file-name">${truncateFileName(file.name)}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                `;
                previewDiv.appendChild(deleteBtn);
            }
            
            previewContainer.appendChild(previewDiv);
        }
    });

    function removeFileFromList(index) {
        const input = document.getElementById('assignmentFiles');
        const files = Array.from(input.files);
        files.splice(index, 1);
        const dataTransfer = new DataTransfer();
        files.forEach(file => dataTransfer.items.add(file));
        input.files = dataTransfer.files;
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function truncateFileName(name) {
        if (name.length > 15) {
            return name.substring(0, 10) + '...' + name.split('.').pop();
        }
        return name;
    }

    function deleteItem(event, id) {
        event.stopPropagation();
        if (confirm(`Are you sure you want to delete submission ${id}? This cannot be undone.`)) {
            showToast(`Deleting submission ${id}...`, 'info');
            setTimeout(() => {
                event.target.closest('.history-item').style.opacity = '0';
                setTimeout(() => {
                    event.target.closest('.history-item').remove();
                    showToast(`Submission ${id} deleted`, 'error');
                }, 300);
            }, 800);
        }
    }

    function showToast(message, type = 'info', duration = 3000) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type}`;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, duration);
    }

    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const files = document.getElementById('assignmentFiles').files;
        if (files.length === 0) {
            showToast('Please select at least one file.', 'error');
            return;
        }
        
        showToast(`Uploading ${files.length} file(s)...`, 'info', 5000);
        
        try {
            const formData = new FormData();
            const timestamp = Date.now();
            
            Array.from(files).forEach(file => {
                formData.append('files', file);
                formData.append('original_names', file.name);
                formData.append('timestamps', timestamp.toString());
            });

            const uploadResponse = await fetch('/upload_assignments', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                throw new Error(`Server responded with ${uploadResponse.status}`);
            }

            const uploadData = await uploadResponse.json();
            

            displayUploadResults(uploadData);
            showToast(`${files.length} file(s) uploaded successfully!`, 'success');
            
            showToast('Starting matching process...', 'info', 3000);
            
            const matchResponse = await fetch('/match_assignments', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    timestamp: timestamp,
                    file_count: files.length
                })
            });
            
            if (!matchResponse.ok) {
                throw new Error(`Matching failed with status ${matchResponse.status}`);
            }

            const matchData = await matchResponse.json();
            console.log("match result===",matchData)
            const resultsDiv = document.getElementById('matchingResults');
            resultsDiv.innerHTML = '<h4><i class="fas fa-check-circle"></i> All matches were completed successfully.</h4>';           
            if(files.length===1){
                displayMatchingResults(matchData.data,1);
            }
            else displayMatchingResults(matchData.data);
            console.log("matchdata==",matchData)
            showToast(`Matching completed for ${files.length} file(s)!`, 'success');

        } catch (error) {
            console.error('Error:', error);
            showToast(`Operation failed: ${error.message}`, 'error');
        }
    });

    function displayUploadResults(data) {
        const resultsDiv = document.getElementById('uploadResults');
        resultsDiv.innerHTML = '<h4><i class="fas fa-check-circle"></i> Uploaded successfully</h4>';           
    }

    function generateAndDownloadPDF(result) {
        const pdfContainer = document.createElement('div');
        pdfContainer.id = 'pdf-container-' + Date.now();
        pdfContainer.style.padding = '20px';
        pdfContainer.style.visibility = 'visible';
        pdfContainer.style.position = 'static';
        pdfContainer.innerHTML=``;
        pdfContainer.innerHTML = `
        <style>
                
            </style>
            <h1 class="pdf-header">Prufia Matching Report</h1>
            <div>
                <h2 class="pdf-title">${result.filename}</h2>
                <table class="pdf-table">
                    <tr>
                        <th>Name or Alias:</th>
                        <td>${result.name_or_alias}</td>
                    </tr>
                    <tr>
                        <th>Teacher:</th>
                        <td>${result.teacher}</td>
                    </tr>
                    <tr>
                        <th>Semester:</th>
                        <td>${result.semester}</td>
                    </tr>
                    <tr>
                        <th>Match Score:</th>
                        <td>
                            <span class="score-badge" style="background: ${getFlagColor(result.flag)};">
                                ${result.score}% (${result.label})
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <th>Date Analyzed:</th>
                        <td>${result.time}</td>
                    </tr>
                </table>
            </div>
            <div class="pdf-footer">
                <p>Generated by Prufia Teacher Dashboard</p>
                <p>Confidence Score: ${getConfidenceDescription(result.score)}</p>
            </div>
        `;

        showToast('Preparing PDF report...', 'info', 2000);
        document.body.appendChild(pdfContainer);
        setTimeout(() => {
            html2pdf().set({
                html2canvas: {
                    scale: 2,
                    logging: false,
                    useCORS: true,
                    letterRendering: true,
                    onclone: function(clonedDoc) {
                        clonedDoc.getElementById(pdfContainer.id).style.visibility = 'visible';
                    }
                },
                jsPDF: {
                    unit: 'mm',
                    format: 'a4',
                    orientation: 'portrait'
                }
            })
            .from(pdfContainer)
            .save(`Prufia_Report_${result.filename.replace(/\.[^/.]+$/, "")}.pdf`)
            .then(() => {
                document.body.removeChild(pdfContainer);
                showToast('PDF downloaded successfully!', 'success');
            })
            .catch(err => {
                console.error('PDF error:', err);
                document.body.removeChild(pdfContainer);
                showToast('PDF generation failed', 'error');
            });
        }, 300);
    }

    function getFlagColor(flag) {
        switch(flag) {
            case 'green': return '#e8f5e9';
            case 'yellow': return '#fff8e1';
            case 'red': return '#ffebee';
            default: return '#f5f7fa';
        }
    }
    function getConfidenceDescription(score) {
        if (score >= 85) return 'Strong Match (Authored by student)';
        if (score >= 70) return 'Likely Match (May include AI tools or human help)';
        return 'Probable Mismatch (Unlikely to be student\'s work)';
    }

    function displayMatchingResults(data, len=null) {
        matchingResults = data;
        const resultsList = document.getElementById('resultsList');
        const resultsCount = document.getElementById('resultsCount');
        const studentFilter = document.getElementById('studentFilter');
        const fileFilter = document.getElementById('fileFilter');
        const confidenceFilter = document.getElementById('confidenceFilter');
        const singleMatchContainer = document.getElementById('singleMatch');
        
        resultsList.innerHTML = '';
        if (singleMatchContainer) singleMatchContainer.innerHTML = '';
        
        resultsCount.textContent = `${data.length} ${data.length === 1 ? 'match' : 'matches'}`;
        
        populateFilterOptions(data, 'name_or_alias', studentFilter);
        populateFilterOptions(data, 'filename', fileFilter);
        
        if (data.length === 0) {
            resultsList.innerHTML = '<div class="no-results">No matching results found</div>';
            return;
        }
        
        data.forEach(result => {
            const resultItem = document.createElement('li');
            resultItem.className = len === 1 ? 'single-result-container' : 'result-item';
            if (len === 1) {
                resultItem.style.cssText = `
                    list-style-type: none; 
                    border: 2px solid #4a89dc;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    background: #f8f9fa;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                `;
                resultItem.onmouseenter = () => {
                    resultItem.style.boxShadow = '0 4px 15px rgba(0,0,0,0.15)';
                    resultItem.style.transform = 'translateY(-2px)';
                };
                resultItem.onmouseleave = () => {
                    resultItem.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
                    resultItem.style.transform = 'translateY(0)';
                };
            }
            resultItem.dataset.name_or_alias = result.name_or_alias;
            resultItem.dataset.filename = result.filename;
            resultItem.dataset.flag = result.flag;
            
            resultItem.innerHTML = `
                <div class="result-info">
                    <div class="result-title">
                        <span>${result.filename}</span>
                        <span class="result-flag flag-${result.flag}">${result.label}</span>
                    </div>
                    <div class="result-meta">
                        <div class="result-meta-item">
                            <i class="fas fa-user-graduate"></i>
                            <span>Name or Alias: ${result.name_or_alias}</span>
                        </div>
                        <div class="result-meta-item">
                            <i class="fas fa-clock"></i>
                            <span>${result.time}</span>
                        </div>
                        <div class="result-meta-item">
                            <i class="fas fa-percentage"></i>
                            <span>Score: ${result.score}%</span>
                        </div>
                    </div>
                    <div class="compact-metrics">
                        <div class="metric-pill" data-tooltip="Sentence Variation: Low">
                            <span class="metric-label">SLV</span>
                            <span class="metric-value">${
                                result.stylometrics?.sentence_len?.std_dev<2.5 ? 'LOW' : 
                                result.stylometrics?.sentence_len?.std_dev>25?'HIGH' : 'MEDIUM'}</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Vocab Entropy: ${
                                result.stylometrics?.vocabulary_entropy?.assess_text_analysis?.ai_score}%">
                            <span class="metric-label">VE</span>
                            <span class="metric-value">${
                                result.stylometrics?.vocabulary_entropy?.assess_text_analysis?.ai_score}%</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Punctuation: Pass">
                            <span class="metric-label">PR</span>
                            <span class="metric-value">${
                                result.stylometrics?.punctual?.assess_text_analysis?.ai_score<50?'✓':'✗'}</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Passive Voice: ${
                                result.stylometrics?.passiv?.assess_text_analysis?.ai_score} %">
                            <span class="metric-label">PV</span>
                            <span class="metric-value">${
                                result.stylometrics?.passiv?.assess_text_analysis?.ai_score} %</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Phrase Reuse: ${result.stylometrics?.repeated?.assess_text_analysis?.ai_score}">
                            <span class="metric-label">RE</span>
                            <span class="metric-value">${result.stylometrics?.repeated?.assess_text_analysis?.ai_score}</span>
                        </div>
                        <div class="metric-pill" data-tooltip="PGFI: 18%">
                            <span class="metric-label">AI</span>
                            <span class="metric-value">${result.stylometrics.pgfi.gpt_phrases.len==0?"HUMAN":"AI"}</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Consistency: ${result.stylometrics.openclose.ass_analysis.error?  result.stylometrics.openclose.scoring_notes.requirements :  result.stylometrics.openclose.estimated_human_reference.expected_consistency}">
                            <span class="metric-label">OC</span>
                            <span class="metric-value">${result.stylometrics.openclose.ass_analysis.error?  result.stylometrics.openclose.scoring_notes.requirements :  result.stylometrics.openclose.estimated_human_reference.expected_consistency}</span>
                        </div>
                    </div>
                </div>
                <div class="result-actions">
                    <button class="action-btn view-btn" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="action-btn download-btn" title="Download Report" data-report-id="${result.id || result.filename}">
                        <i class="fas fa-download"></i>
                    </button>
                </div>
            `;
            
            
            // Add click handler for view button
            resultItem.querySelector('.view-btn').addEventListener('click', () => {
                showResultDetails(result);
            });
            
            // Add click handler for download button
            resultItem.querySelector('.download-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                generateAndDownloadPDF(result);
            });
            if (len === 1) {
                singleMatchContainer.appendChild(resultItem);
            } else {
                resultsList.appendChild(resultItem);
            }
        });
    }
    
    function populateFilterOptions(data, property, selectElement) {
        const uniqueValues = [...new Set(data.map(item => item[property]))];
        uniqueValues.forEach(value => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            selectElement.appendChild(option);
        });
    }
    
    function showResultDetails(result) {
        const detailContent = document.getElementById('detailContent');
        
        detailContent.innerHTML = `
            <div class="report-header">
                <h3>Document Analysis Report</h3>                    
            </div>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="metadata-label">Filename:</div>
                    <div class="metdata-value">${result.filename}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Name or Alias:</div>
                    <div class="metdata-value">${result.name_or_alias}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Teacher:</div>
                    <div class="metdata-value">${result.teacher}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Semester:</div>
                    <div class="metdata-value">${result.semester}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Match Score:</div>
                    <div class="metdata-value">
                        <span class="result-flag flag-${result.flag}">${result.score}%</span>
                    </div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Status:</div>
                    <div class="metdata-value">
                        <span class="result-flag flag-${result.flag}">${result.label}</span>
                    </div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Date:</div>
                    <div class="metdata-value">${result.time}</div>
                </div>
            </div>
                        
            <div class="metric-section">
                <h4 class="section-title">Stylometrics</h4>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon"><i class="fas fa-ruler-combined"></i></div>
                            <h5>Sentence Variation</h5>
                        </div>
                        <div class="metric-value">${
                                result.stylometrics?.sentence_len?.std_dev<2.5 ? 'LOW' : 
                                result.stylometrics?.sentence_len?.std_dev>25?'HIGH' : 'MEDIUM'}</div>
                        <div class="metric-description">${result.stylometrics?.sentence_len?.assessment?.ai_score}-${result.stylometrics?.sentence_len?.assessment?.assessment}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon"><i class="fas fa-language"></i></div>
                            <h5>Vocabulary Entropy</h5>
                        </div>
                        <div class="metric-value">${
                                result.stylometrics?.vocabulary_entropy?.assess_text_analysis?.ai_score}%</div>
                        <div class="metric-description">${
                                result.stylometrics?.vocabulary_entropy?.assess_text_analysis?.assessment}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon"><i class="fas fa-stopwatch"></i></div>
                            <h5>Punctuation Rhythm</h5>
                        </div>
                        <div class="metric-value">${
                                result.stylometrics?.punctual?.assess_text_analysis?.ai_score}</div>
                        <div class="metric-description">${
                                result.stylometrics?.punctual?.assess_text_analysis?.assessment}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon"><i class="fas fa-voice"></i></div>
                            <h5>Passive Voice</h5>
                        </div>
                        <div class="metric-value">${
                                result.stylometrics?.passiv?.assess_text_analysis?.ai_score} %</div>
                        <div class="metric-description">${
                                result.stylometrics?.passiv?.assess_text_analysis?.assessment}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon"><i class="fas fa-redo-alt"></i></div>
                            <h5>Phrase Reuse</h5>
                        </div>
                        <div class="metric-value">${result.stylometrics?.repeated?.assess_text_analysis?.ai_score}</div>
                        <div class="metric-description">${result.stylometrics?.repeated?.assess_text_analysis?.assessment}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon"><i class="fas fa-robot"></i></div>
                            <h5>PGFI (AI Mimicry)</h5>
                        </div>
                        <div class="metric-value">${result.stylometrics.pgfi.gpt_phrases.len==0?"HUMAN":"AI"}</div>
                        <div class="metric-description">Variance is ${result.stylometrics.pgfi.semantic_variance}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon"><i class="fas fa-book-open"></i></div>
                            <h5>Structure Consistency</h5>
                        </div>
                        <div class="metric-value">${result.stylometrics?.openclose?.ass_analysis?.error}</div>
                        <div class="metric-description">
                            ${result.stylometrics.openclose.ass_analysis.error?  result.stylometrics.openclose.scoring_notes.requirements :  result.stylometrics.openclose.estimated_human_reference.expected_consistency}
                        </div>
                    </div>
                </div>
            </div>
                
            <div class="detail-row">
                <div class="">
                    <button class="primary-button download-detail-btn" 
                    style="padding: 5px 10px; font-size: 0.9rem;"
                    data-report-id="${result.id || result.filename}">
                        <i class="fas fa-download"></i> Download Report
                    </button>
                </div>
            </div>     
        `;
        detailContent.querySelector('.download-detail-btn').addEventListener('click', () => {
            generateAndDownloadPDF(result);
        });
    }
    
    function filterResults() {
        const studentFilter = document.getElementById('studentFilter').value;
        const fileFilter = document.getElementById('fileFilter').value;
        const confidenceFilter = document.getElementById('confidenceFilter').value;
        
        const filteredResults = matchingResults.filter(result => {
            const studentMatch = !studentFilter || result.name_or_alias === studentFilter;
            const fileMatch = !fileFilter || result.filename === fileFilter;
            const confidenceMatch = !confidenceFilter || result.flag === confidenceFilter;
            
            return studentMatch && fileMatch && confidenceMatch;
        });
        
        const resultsList = document.getElementById('resultsList');
        resultsList.innerHTML = '';
        
        resultsCount.textContent = `${filteredResults.length} ${filteredResults.length === 1 ? 'match' : 'matches'}`;

        if (filteredResults.length === 0) {
            resultsList.innerHTML = '<div class="no-results">No matching results found</div>';
            return;
        }
        
        filteredResults.forEach(result => {
            const resultItem = document.createElement('li');
            resultItem.className = 'result-item';
            resultItem.dataset.name_or_alias = result.name_or_alias;
            resultItem.dataset.filename = result.filename;
            resultItem.dataset.flag = result.flag;
            
            resultItem.innerHTML = `
                <div class="result-info">
                    <div class="result-title">
                        <span>${result.filename}</span>
                        <span class="result-flag flag-${result.flag}">${result.label}</span>
                    </div>
                    <div class="result-meta">
                        <div class="result-meta-item">
                            <i class="fas fa-user-graduate"></i>
                            <span>Name or Alias: ${result.name_or_alias}</span>
                        </div>
                        <div class="result-meta-item">
                            <i class="fas fa-clock"></i>
                            <span>${result.time}</span>
                        </div>
                        <div class="result-meta-item">
                            <i class="fas fa-percentage"></i>
                            <span>Score: ${result.score}%</span>
                        </div>
                    </div>
                    <div class="compact-metrics">
                        <div class="metric-pill" data-tooltip="Sentence Variation: Low">
                            <span class="metric-label">SLV</span>
                            <span class="metric-value">${
                                result.stylometrics?.sentence_len?.std_dev<2.5 ? 'LOW' : 
                                result.stylometrics?.sentence_len?.std_dev>25?'HIGH' : 'MEDIUM'}</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Vocab Entropy: ${
                                result.stylometrics?.vocabulary_entropy?.assess_text_analysis?.ai_score} %">
                            <span class="metric-label">VE</span>
                            <span class="metric-value">${
                                result.stylometrics?.vocabulary_entropy?.assess_text_analysis?.ai_score} %</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Punctuation: Pass">
                            <span class="metric-label">PR</span>
                            <span class="metric-value">${
                                result.stylometrics?.punctual?.assess_text_analysis?.ai_score}-${
                                result.stylometrics?.punctual?.assess_text_analysis?.ai_score<50?'✓':'X'}</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Passive Voice: ${
                                result.stylometrics?.passiv?.assess_text_analysis?.ai_score} %">
                            <span class="metric-label">PV</span>
                            <span class="metric-value">${
                                result.stylometrics?.passiv?.assess_text_analysis?.assessment} %</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Phrase Reuse: ${result.stylometrics?.repeated?.assess_text_analysis?.ai_score}">
                            <span class="metric-label">RE</span>
                            <span class="metric-value">${result.stylometrics?.repeated?.assess_text_analysis?.ai_score}</span>
                        </div>
                        <div class="metric-pill" data-tooltip="PGFI: 18%">
                            <span class="metric-label">AI</span>
                            <span class="metric-value">${result.stylometrics.pgfi.gpt_phrases.len==0?"HUMAN":"AI"} %</span>
                        </div>
                        <div class="metric-pill" data-tooltip="Consistency: ${result.stylometrics.openclose.ass_analysis.error?  result.stylometrics.openclose.scoring_notes.requirements :  result.stylometrics.openclose.estimated_human_reference.expected_consistency}">
                            <span class="metric-label">OC</span>
                            <span class="metric-value">${result.stylometrics.openclose.ass_analysis.error?  result.stylometrics.openclose.scoring_notes.requirements :  result.stylometrics.openclose.estimated_human_reference.expected_consistency}</span>
                        </div>
                    </div>
                </div>
                <div class="result-actions">
                    <button class="action-btn view-btn" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="action-btn download-btn" title="Download Report">
                        <i class="fas fa-download"></i>
                    </button>
                </div>
            `;
            
            resultsList.appendChild(resultItem);
            
            resultItem.querySelector('.view-btn').addEventListener('click', () => {
                showResultDetails(result);
            });
        });
    }
    
    document.getElementById('studentFilter').addEventListener('change', filterResults);
    document.getElementById('fileFilter').addEventListener('change', filterResults);
    document.getElementById('confidenceFilter').addEventListener('change', filterResults);
    document.querySelector('.filter-reset').addEventListener('click', function() {
        document.getElementById('studentFilter').value = '';
        document.getElementById('fileFilter').value = '';
        document.getElementById('confidenceFilter').value = '';
        filterResults();
        showToast('All filters have been reset', 'success');
    });
    document.getElementById('resetMatch').addEventListener('click', function() {
        document.getElementById('assignmentFiles').value = '';
        const filePreviews = document.getElementById('filePreviews');
        filePreviews.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-file-alt"></i>
                <p>No files selected</p>
            </div>
        `;
        const originalText = this.innerHTML;
        this.innerHTML = '<i class="fas fa-check"></i> Cleared';
        setTimeout(() => {
            this.innerHTML = originalText;
        }, 1500);

        displayMatchingResults([]);
    });