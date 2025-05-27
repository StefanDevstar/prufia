const socket = io({
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    debug: true,
    query: {
        'user-type': 'admin' 
    }
});
const socketStatus = document.getElementById('socket-status');
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
socket.on('connect', () => {
    socketStatus.textContent = 'Connected ✓';
    socketStatus.className = 'badge bg-success';
    socket.emit('join-admin-room');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    socketStatus.textContent = 'Disconnected ✗';
    socketStatus.className = 'badge bg-danger';
});

socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
    socketStatus.textContent = 'Connection Error';
    socketStatus.className = 'badge bg-warning text-dark';
});

document.getElementById('downloadCsvBtn').addEventListener('click', function() {
    fetch('/api/submissions')
        .then(response => response.json())
        .then(data => {
            const headers = ['Student ID', 'Name', 'Submitted At', 'Status', 'Score'];
            const rows = data.map(item => [
                item.student_id,
                item.name_or_alias,
                item.created_at,
                item.status,
                item.score || ''
            ]);
            const csvContent = [headers, ...rows]
                .map(row => row.join(','))
                .join('\n');
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.setAttribute('href', url);
            link.setAttribute('download', 'student_submissions_' + new Date().toISOString().slice(0, 10) + '.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        })
        .catch(error => {
            console.error('Error downloading CSV:', error);
            alert('Failed to download CSV');
        });
});
document.querySelectorAll('.generate-passcode-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const studentId = this.getAttribute('data-student-id');
        document.getElementById('confirmStudentId').textContent = studentId;
        const confirmModal = new bootstrap.Modal(document.getElementById('confirmGenerateModal'));
        confirmModal.show();
        document.getElementById('confirmGenerateBtn').dataset.studentId = studentId;
        document.getElementById('confirmGenerateBtn').dataset.originalButton = this;
    });
});
document.getElementById('confirmGenerateBtn').addEventListener('click', async function() {
    const studentId = this.dataset.studentId;
    const originalButton = this.dataset.originalButton;
    const passcodeCell = document.getElementById(`passcode-${studentId}`);
    const statusCell = document.getElementById(`use-status-${studentId}`);
    const confirmModal = bootstrap.Modal.getInstance(document.getElementById('confirmGenerateModal'));
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
            const modal = new bootstrap.Modal(document.getElementById('passcodeModal'));
            modal.show();
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
});

document.getElementById('resetAllBtn')?.addEventListener('click', async function() {
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
        showNotification(`Successfully reset ${data.affected_rows || 0} passcodes`);
        setTimeout(() => {
            window.location.reload();
        }, 1500);
        
    } catch (error) {
        console.error('Reset error:', error);
        showNotification(error.message || 'Error resetting passcodes', 'danger');
    } finally {
        btn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Reset All Passcodes';
        btn.disabled = false;
    }
});
function showNotificationreset(message, type = 'success') {
    const toastEl = document.getElementById('notificationToast');
    const toastBody = document.getElementById('toastMessage');
    if (toastEl && toastBody) {
        const toastHeader = toastEl.querySelector('.toast-header');
        toastHeader.className = `toast-header bg-${type} text-white`;
        toastBody.textContent = message;
        const toast = new bootstrap.Toast(toastEl);
        toast.show();
    }
}

function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]')?.content || '';
}

document.getElementById('copyPasscodeBtn')?.addEventListener('click', function() {
    const passcode = document.getElementById('generatedPasscode').textContent;
    navigator.clipboard.writeText(passcode).then(() => {
        this.innerHTML = '<i class="bi bi-check-circle"></i> Copied!';
        setTimeout(() => {
            this.innerHTML = '<i class="bi bi-clipboard"></i> Copy';
        }, 2000);
    });
});

function showNotification(message) {
    const toastEl = document.getElementById('notificationToast');
    const toastBody = document.getElementById('toastMessage');
    
    if (toastEl && toastBody) {
        toastBody.textContent = message;
        const toast = new bootstrap.Toast(toastEl);
        toast.show();
    }
}
    