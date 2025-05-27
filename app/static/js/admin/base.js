 const socket = io({
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    query: { 'user-type': 'admin' }
});

const socketStatus = document.getElementById('socket-status');
socket.on('connect', () => {
    socketStatus.textContent = 'Connected ✓';
    socketStatus.className = 'badge bg-success';
    socket.emit('join-admin-room');
});

socket.on('disconnect', () => {
    socketStatus.textContent = 'Disconnected ✗';
    socketStatus.className = 'badge bg-danger';
});

socket.on('connect_error', (error) => {
    socketStatus.textContent = 'Connection Error';
    socketStatus.className = 'badge bg-warning text-dark';
});

socket.on('student-login', (data) => {
    const row = document.querySelector(`tr[data-student-id="${data.student_id}"]`);
    if (!row) return;
    
    const statusCell = row.querySelector(`#use-status-${data.student_id}`);
    if (statusCell) {
        statusCell.innerHTML = `
            <span class="badge bg-danger">Used</span>
            <small class="text-muted">${new Date().toLocaleTimeString()}</small>
        `;
    }
    
    const genBtn = row.querySelector('.generate-passcode-btn');
    if (genBtn) genBtn.disabled = true;
    
    showNotification(`Student ${data.student_id} logged in at ${new Date().toLocaleTimeString()}`);
});
function showNotification(message, type = 'primary') {
    const toastEl = document.getElementById('notificationToast');
    const toastBody = document.getElementById('toastMessage');
    
    if (toastEl && toastBody) {
        const toastHeader = toastEl.querySelector('.toast-header');
        toastHeader.className = `toast-header bg-${type} text-white`;
        toastBody.textContent = message;
        new bootstrap.Toast(toastEl).show();
    }
}

document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const studentId = this.getAttribute('data-id');
        console.log("studentid===",studentId)
        fetch(`/view_submission/${studentId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('submissionDetails').innerHTML = 
                        `<p class="text-danger">${data.error}</p>`;
                } else {
                    let details = '';
                    for (const [key, value] of Object.entries(data)) {
                        details += `<p><strong>${key.replace('_', ' ').toUpperCase()}:</strong> </p><p style="padding-left: 10px;">${value}</p>`;
                    }
                    document.getElementById('submissionDetails').innerHTML = details;
                }
            });
    });
});
function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]')?.content || '';
}
function getScoreClass(score) {
    if (score >= 85) return 'green';
    if (score >= 65) return 'yellow';
    return 'red';
}

function getScoreLabel(score) {
    if (score >= 85) return 'Strong Match';
    if (score >= 65) return 'Needs Review';
    return 'Possible Mismatch';
}
function loadContent(url) {
    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.text();
        })
        .then(html => {
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = html;
            
            const scripts = tempDiv.querySelectorAll('script');
            scripts.forEach(script => script.remove());
            
            document.querySelector('.main-content').innerHTML = tempDiv.innerHTML;
            
            if (url.includes('submissions')) {
                initSubmissionsPage();
            } else if (url.includes('passcode')) {
                initPasscodePage();
            }
        })
        .catch(error => {
            console.error('Error loading content:', error);
            showNotification('Failed to load content', 'danger');
        });
}

function initSubmissionsPage() {
    console.log('Submissions page initialized');
}

window.initPasscodePage = function() {
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('generate-passcode-btn')) {
            const studentId = e.target.getAttribute('data-student-id');
            document.getElementById('confirmStudentId').textContent = studentId;
            const confirmModal = new bootstrap.Modal(document.getElementById('confirmGenerateModal'));
            confirmModal.show();
            document.getElementById('confirmGenerateBtn').onclick = async function() {
                const originalButton = e.target;
                const passcodeCell = document.getElementById(`passcode-${studentId}`);
                const statusCell = document.getElementById(`use-status-${studentId}`);
                confirmModal.hide();
                originalButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Generating...';
                originalButton.disabled = true;
                try {
                    const response = await fetch(`/generate-passcode/${studentId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    const data = await response.json();
                    const now = new Date();
                    
                    if (data.status === 'success') {
                        passcodeCell.innerHTML = `
                            <code>${data.passcode}</code>
                            <div class="text-success small">Generated: ${now.toLocaleString()}</div>
                        `;
                        
                        if (statusCell) {
                            statusCell.innerHTML = `
                                <span class="badge bg-success">Active</span>
                                <small class="text-muted">Not used yet</small>
                            `;
                        }
                        
                        document.getElementById('studentName').textContent = studentId;
                        document.getElementById('generatedPasscode').textContent = data.passcode;
                        document.getElementById('passcodeExpires').textContent = 
                            new Date(data.expires_at).toLocaleString();
                        
                        new bootstrap.Modal(document.getElementById('passcodeModal')).show();
                    } else {
                        showNotification(`Error: ${data.message || 'Failed to generate passcode'}`);
                    }
                } catch (error) {
                    console.error('Error:', error);
                    showNotification('Failed to generate passcode');
                } finally {
                    originalButton.innerHTML = '<i class="bi bi-key"></i> Generate';
                    originalButton.disabled = false;
                }
            };
        }
    });
    const resetBtn = document.getElementById('resetAllBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', async function() {
            const btn = this;
            
            if (!confirm('Are you sure you want to reset ALL passcodes?')) {
                return;
            }
            
            btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Resetting...';
            btn.disabled = true;
            
            try {
                const response = await fetch('/reset', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                    }
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.message || 'Failed to reset passcodes');
                }
                
                showNotification(`Successfully reset ${data.affected_rows || 0} passcodes`, 'success');
                setTimeout(() => {
                    loadContent('/passcode');
                }, 1500);
                
            } catch (error) {
                console.error('Reset error:', error);
                showNotification(error.message || 'Error resetting passcodes', 'danger');
            } finally {
                btn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Reset All Passcodes';
                btn.disabled = false;
            }
        });
    }
    const copyBtn = document.getElementById('copyPasscodeBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            const passcode = document.getElementById('generatedPasscode').textContent;
            navigator.clipboard.writeText(passcode).then(() => {
                this.innerHTML = '<i class="bi bi-check-circle"></i> Copied!';
                setTimeout(() => {
                    this.innerHTML = '<i class="bi bi-clipboard"></i> Copy';
                }, 2000);
            });
        });
    }
};
document.addEventListener('DOMContentLoaded', () => {
    loadContent('/passcode'); 
});
document.addEventListener('click', function(e) {
    if (e.target.closest('.view-btn')) {
        const btn = e.target.closest('.view-btn');
        const studentId = btn.getAttribute('data-id');
        console.log(studentId)
        fetch(`/view_submission_admin/${studentId}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw err; });
                }
                return response.json();
            })
            .then(data => {
                const detailsContainer = document.getElementById('submissionDetails');
                if (data.error) {
                    if (data.error.includes("already been approved")) {
                        detailsContainer.innerHTML = `
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle me-2"></i>
                                ${data.error}
                            </div>
                        `;
                    } else {
                        detailsContainer.innerHTML = `
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                ${data.error}
                            </div>
                        `;
                    }
                } else {
                    let details = '';
                    if (data.status === 2) {
                        details += `
                            <div class="alert alert-success mb-3">
                                <i class="fas fa-check-circle me-2"></i>
                                This submission has been approved
                            </div>
                        `;
                    }
                    for (const [key, value] of Object.entries(data)) {
                        if (key === 'status') continue; 
                        
                        const displayKey = key.replace(/_/g, ' ').toUpperCase();
                        const displayValue = value || 'Not available';
                        
                        details += `
                            <div class="mb-2">
                                <strong>${displayKey}:</strong>
                                <div class="ms-3">${displayValue}</div>
                            </div>
                        `;
                    }
                    
                    detailsContainer.innerHTML = details;
                }
            })
            .catch(error => {
                document.getElementById('submissionDetails').innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        ${error.error || 'This file was already approved, so it has been removed from our database.'}
                    </div>
                `;
            });
    }
    
    if (e.target.closest('.approve-btn')) {
        const btn = e.target.closest('.approve-btn');
        const studentId = btn.dataset.id;
        const nameOrAlias = btn.dataset.name_or_alias;
        const createdAt = btn.dataset.created_at;
        document.getElementById('instructorFeedback').textContent = nameOrAlias;
        document.getElementById('submissionDateDisplay').textContent = createdAt;
        document.getElementById('resubmitStudentId').value=studentId
        document.getElementById('approveSwitch').checked=studentId
        const row = btn.closest('tr');
        if (row) {
            const rowId = row.id.replace('baseline-', '');
        }
    }
});

document.addEventListener('submit', function(e) {
    if (e.target.id === 'approveForm') {
        e.preventDefault();                
        const base_id=document.getElementById('resubmitStudentId').value
        const is_approved=document.getElementById('approveSwitch').checked
        if (base_id) {
            fetch('/approve-endpoint', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    base_id: base_id,
                    is_approved:is_approved
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                const modal = bootstrap.Modal.getInstance(document.getElementById('approveModal'));
                modal.hide();
                
                showNotification("Approved successfully!", "info")
                if (data.success) {
                    const row = document.getElementById(`baseline-${base_id}`);
                    if (row) {
                        const statusCell = row.querySelector('#status-' + base_id);
                        if (statusCell && is_approved) {
                            statusCell.innerHTML = `
                                <span class="badge bg-success status-badge">Approved</span>
                                <small class="text-muted d-block">${new Date().toLocaleString()}</small>
                            `;
                        }
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error processing request: ' + error.message);
            })
            .finally(() => {
                submitBtn.disabled = false;
                document.getElementById('submitSpinner').classList.add('d-none');
                document.getElementById('submitButtonText').textContent = 'Submit Approval';
            });
        }
    }
});
