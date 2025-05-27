$(document).ready(function() {
    let matchingResults = [];    
    const $uploadForm = $('#uploadForm');
    let uploadedFilename = '';
    
    function toggleDetails(button) {
        const $item = $(button).closest('.history-item');
        $item.toggleClass('expanded');
        $(button).toggleClass('rotate-180');
    }

    // Initialize socket.io
    const socket = io({
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        query: { 'user-type': 'admin' }
    });

    // Socket status management
    const $socketStatus = $('#socket-status');
    socket.on('connect', () => {
        console.log("I am joined");
        socket.emit('join-admin-room');
    });

    socket.on('disconnect', () => {
        // Handle disconnect
    });

    socket.on('connect_error', (error) => {
        // Handle connection error
    });

    // Handle student login events
    socket.on('progress', (data) => {
        const $progressBar = $('#progress-bar');
        const $statusNote = $('#progress-status');
        $progressBar.css('width', `${data.value}%`);
        $statusNote.text(`${data.value}% - ${data.func_name} was analyzed.`);
   
        if (data.value >= 99) {
            setTimeout(() => {
                $progressBar.css('width', '0%');
                $statusNote.text('Completed');
            }, 2000);
        }  
    });

    // Dropdown functionality
    const $dropdownInput = $('.dropdown-input');
    const $dropdownList = $('.dropdown-list');
    const $checkboxes = $('.dropdown-item input[type="checkbox"]');
    const $selectAll = $('#select-all');
    
    function initialize() {
        $checkboxes.prop('checked', true);
        updateSelectedOptions();
    }
    
    function updateSelectedOptions() {
        const selectedOptions = $checkboxes.not($selectAll).filter(':checked').map(function() {
            return $(this).val();
        }).get();
        
        $dropdownInput.val(selectedOptions.join(', '));
        
        const allChecked = $checkboxes.not($selectAll).length === $checkboxes.not($selectAll).filter(':checked').length;
        $selectAll.prop('checked', allChecked);
        $selectAll.prop('indeterminate', !allChecked && selectedOptions.length > 0);
    }
    
    $dropdownInput.on('click', function() {
        $dropdownList.toggleClass('show');
    });
    
    $(document).on('click', function(event) {
        if (!$(event.target).closest('.dropdown-container').length) {
            $dropdownList.removeClass('show');
        }
    });
    
    $checkboxes.on('change', function() {
        if (this === $selectAll[0]) {
            $checkboxes.not($selectAll).prop('checked', $selectAll.prop('checked'));
        }
        updateSelectedOptions();
    });
    
    initialize();

    // File preview functionality
    $('#clearSelectionBtn').on('click', function() {
        $('#assignmentFiles').val('');
        $('#filePreviews').html(`
            <div class="empty-state">
                <i class="fas fa-file-alt"></i>
                <p>No files selected</p>
            </div>
        `);
        const $this = $(this);
        const originalText = $this.html();
        $this.html('<i class="fas fa-check"></i> Cleared');
        setTimeout(() => {
            $this.html(originalText);
        }, 1500);
    });

    $('#assignmentFiles').on('change', function(e) {
        const files = e.target.files;
        const $previewContainer = $('#filePreviews');
        $previewContainer.empty();
        
        if (files.length === 0) return;
        
        $.each(files, function(i, file) {
            const fileExt = file.name.split('.').pop().toLowerCase();
            const $previewDiv = $('<div>').addClass('file-preview').data('file-index', i);
            const $deleteBtn = $('<div>').addClass('delete-preview').html('×');
            
            $deleteBtn.on('click', function(e) {
                e.stopPropagation();
                removeFileFromList(i);
                $previewDiv.remove();
            });
            
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    $previewDiv.html(`
                        <div><i class="fas fa-file-image img-icon file-icon"></i></div>
                        <img src="${e.target.result}" class="preview-image" alt="${file.name}">
                        <div class="file-name">${truncateFileName(file.name)}</div>
                        <div class="file-size">${formatFileSize(file.size)}</div>
                    `).append($deleteBtn);
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
                
                $previewDiv.html(`
                    <i class="fas ${iconClass} file-icon ${extraClass}"></i>
                    <div class="file-name">${truncateFileName(file.name)}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                `).append($deleteBtn);
            }
            
            $previewContainer.append($previewDiv);
        });
    });

    function removeFileFromList(index) {
        const $input = $('#assignmentFiles');
        const files = Array.from($input[0].files);
        files.splice(index, 1);
        const dataTransfer = new DataTransfer();
        files.forEach(file => dataTransfer.items.add(file));
        $input[0].files = dataTransfer.files;
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
                $(event.target).closest('.history-item').css('opacity', '0');
                setTimeout(() => {
                    $(event.target).closest('.history-item').remove();
                    showToast(`Submission ${id} deleted`, 'error');
                }, 300);
            }, 800);
        }
    }

    function showToast(message, type = 'info', duration = 3000) {
        const $toast = $('#toast');
        $toast.text(message)
            .removeClass()
            .addClass(`toast ${type} show`);
        
        setTimeout(() => {
            $toast.removeClass('show');
        }, duration);
    }

    $uploadForm.on('submit', async function(e) {
        alert("ssssssssssssssssssssssssssssssssss")
        e.preventDefault();
        
        const files = $('#assignmentFiles')[0].files;
        if (files.length === 0) {
            showToast('Please select at least one file.', 'error');
            return;
        }
        
        showToast(`Uploading ${files.length} file(s)...`, 'info', 5000);
        
        try {
            const formData = new FormData();
            const timestamp = Date.now();
            
            $.each(files, function(i, file) {
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
            console.log("match result===", matchData);
            const $resultsDiv = $('#matchingResults');
            $resultsDiv.html('<h4><i class="fas fa-check-circle"></i> All matches were completed successfully.</h4>');           
            if (files.length === 1) {
                displayMatchingResults(matchData.data, 1);
            } else {
                displayMatchingResults(matchData.data);
            }
            console.log("matchdata==", matchData);
            showToast(`Matching completed for ${files.length} file(s)!`, 'success');
        } catch (error) {
            console.error('Error:', error);
            showToast(`Operation failed: ${error.message}`, 'error');
        }
    });

    function displayUploadResults(data) {
        $('#uploadResults').html('<h4><i class="fas fa-check-circle"></i> Uploaded successfully</h4>');           
    }

    function generateAndDownloadPDF(result) {
        const $pdfContainer = $('<div>')
            .attr('id', 'pdf-container-' + Date.now())
            .css({
                padding: '20px',
                visibility: 'visible',
                position: 'static'
            })
            .html(`
                <style>
                    .pdf-header {
                        color: #4a6fa5;
                        border-bottom: 2px solid #4a6fa5;
                        padding-bottom: 10px;
                        margin-bottom: 20px;
                    }
                    .pdf-title {
                        color: #2c3e50;
                        margin-top: 20px;
                    }
                    .pdf-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 15px;
                        font-size: 14px;
                    }
                    .pdf-table th, .pdf-table td {
                        padding: 10px;
                        border: 1px solid #ddd;
                        text-align: left;
                    }
                    .pdf-table th {
                        background-color: #f5f7fa;
                        font-weight: bold;
                        width: 30%;
                    }
                    .pdf-footer {
                        margin-top: 30px;
                        font-size: 12px;
                        color: #666;
                        text-align: center;
                    }
                    .score-badge {
                        padding: 4px 12px;
                        border-radius: 12px;
                        font-weight: 500;
                        display: inline-block;
                    }
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
            `);
        
        $('body').append($pdfContainer);
        
        showToast('Preparing PDF report...', 'info', 2000);
        
        setTimeout(() => {
            html2pdf().from($pdfContainer[0]).save(`Prufia_Report_${result.filename.replace(/\.[^/.]+$/, "")}.pdf`)
                .then(() => {
                    $pdfContainer.remove();
                    showToast('PDF downloaded successfully!', 'success');
                })
                .catch(err => {
                    console.error('PDF error:', err);
                    $pdfContainer.remove();
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

    function displayMatchingResults(data, len = null) {
        matchingResults = data;
        const $resultsList = $('#resultsList').empty();
        const $resultsCount = $('#resultsCount');
        const $studentFilter = $('#studentFilter');
        const $fileFilter = $('#fileFilter');
        const $confidenceFilter = $('#confidenceFilter');
        const $singleMatchContainer = $('#singleMatch').empty();
        
        $resultsCount.text(`${data.length} ${data.length === 1 ? 'match' : 'matches'}`);
        
        populateFilterOptions(data, 'name_or_alias', $studentFilter);
        populateFilterOptions(data, 'filename', $fileFilter);
        
        if (data.length === 0) {
            $resultsList.html('<div class="no-results">No matching results found</div>');
            return;
        }
        
        $.each(data, function(i, result) {
            const $resultItem = $('<li>')
                .addClass(len === 1 ? 'single-result-container' : 'result-item')
                .data({
                    name_or_alias: result.name_or_alias,
                    filename: result.filename,
                    flag: result.flag
                });
            
            if (len === 1) {
                $resultItem.css({
                    'list-style-type': 'none',
                    'border': '2px solid #4a89dc',
                    'border-radius': '8px',
                    'padding': '20px',
                    'margin': '20px 0',
                    'background': '#f8f9fa',
                    'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
                    'transition': 'all 0.3s ease'
                });
                
                $resultItem.hover(
                    function() {
                        $(this).css({
                            'box-shadow': '0 4px 15px rgba(0,0,0,0.15)',
                            'transform': 'translateY(-2px)'
                        });
                    },
                    function() {
                        $(this).css({
                            'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
                            'transform': 'translateY(0)'
                        });
                    }
                );
            }
            
            $resultItem.html(`
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
            `);
            
            $resultItem.find('.view-btn').on('click', function() {
                showResultDetails(result);
            });
            
            $resultItem.find('.download-btn').on('click', function(e) {
                e.stopPropagation();
                generateAndDownloadPDF(result);
            });
            
            if (len === 1) {
                $singleMatchContainer.append($resultItem);
            } else {
                $resultsList.append($resultItem);
            }
        });
    }
    
    function populateFilterOptions(data, property, $selectElement) {
        const uniqueValues = [...new Set(data.map(item => item[property]))];
        $selectElement.empty().append('<option value="">All</option>');
        $.each(uniqueValues, function(i, value) {
            $selectElement.append($('<option>').val(value).text(value));
        });
    }
    
    function showResultDetails(result) {
        const $detailContent = $('#detailContent');
        
        $detailContent.html(`
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
        `);
        
        $detailContent.find('.download-detail-btn').on('click', function() {
            generateAndDownloadPDF(result);
        });
    }
    
    function filterResults() {
        const studentFilter = $('#studentFilter').val();
        const fileFilter = $('#fileFilter').val();
        const confidenceFilter = $('#confidenceFilter').val();
        
        const filteredResults = matchingResults.filter(result => {
            const studentMatch = !studentFilter || result.name_or_alias === studentFilter;
            const fileMatch = !fileFilter || result.filename === fileFilter;
            const confidenceMatch = !confidenceFilter || result.flag === confidenceFilter;
            
            return studentMatch && fileMatch && confidenceMatch;
        });
        
        const $resultsList = $('#resultsList').empty();
        
        $('#resultsCount').text(`${filteredResults.length} ${filteredResults.length === 1 ? 'match' : 'matches'}`);

        if (filteredResults.length === 0) {
            $resultsList.html('<div class="no-results">No matching results found</div>');
            return;
        }
        
        $.each(filteredResults, function(i, result) {
            const $resultItem = $('<li>').addClass('result-item')
                .data({
                    'name_or_alias': result.name_or_alias,
                    'filename': result.filename,
                    'flag': result.flag
                })
                .html(`
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
                `);
            
            $resultsList.append($resultItem);
            
            $resultItem.find('.view-btn').on('click', function() {
                showResultDetails(result);
            });
        });
    }
    
    $('#studentFilter, #fileFilter, #confidenceFilter').on('change', filterResults);
    $('.filter-reset').on('click', function() {
        $('#studentFilter, #fileFilter, #confidenceFilter').val('');
        filterResults();
        showToast('All filters have been reset', 'success');
    });
    
    $('#resetMatch').on('click', function() {
        $('#assignmentFiles').val('');
        $('#filePreviews').html(`
            <div class="empty-state">
                <i class="fas fa-file-alt"></i>
                <p>No files selected</p>
            </div>
        `);
        const $this = $(this);
        const originalText = $this.html();
        $this.html('<i class="fas fa-check"></i> Cleared');
        setTimeout(() => {
            $this.html(originalText);
        }, 1500);

        displayMatchingResults([]);
    });
});