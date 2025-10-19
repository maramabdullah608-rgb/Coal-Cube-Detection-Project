// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const resultsSection = document.getElementById('results-section');
const loadingSpinner = document.getElementById('loading-spinner');
const errorAlert = document.getElementById('error-alert');
const errorMessage = document.getElementById('error-message');
const modelStatus = document.getElementById('model-status');
const modelStatusText = document.getElementById('model-status-text');

// Chart instances
let qualityChart = null;
let defectChart = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // File input change event
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Click on upload area
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Check model status on load
    checkModelStatus();
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && validateFile(file)) {
        displayImagePreview(file);
        analyzeImage(file);
    }
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0 && validateFile(files[0])) {
        fileInput.files = files;
        displayImagePreview(files[0]);
        analyzeImage(files[0]);
    }
}

// Validate file
function validateFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        showError('يرجى رفع ملف صورة صالح (JPG, PNG, GIF, BMP)');
        return false;
    }
    
    if (file.size > maxSize) {
        showError('يجب أن يكون حجم الملف أقل من 16 ميجابايت');
        return false;
    }
    
    return true;
}

// Display image preview
function displayImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        imagePreview.classList.remove('d-none');
    };
    
    reader.readAsDataURL(file);
}

// Clear image
function clearImage() {
    fileInput.value = '';
    imagePreview.classList.add('d-none');
    hideResults();
}

// Analyze image
function analyzeImage(file) {
    showLoading();
    hideResults();
    hideError();
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        hideLoading();
        showError('خطأ في الشبكة: ' + error.message);
        console.error('Analysis error:', error);
    });
}

// Display results
function displayResults(data) {
    // Display uploaded image
    document.getElementById('result-image').innerHTML = `
        <img src="${data.image}" class="img-fluid rounded shadow" style="max-height: 250px;">
        <p class="text-muted mt-2">الصورة المرفوعة</p>
    `;
    
    // Display quality results
    const quality = data.quality_prediction;
    const qualityElement = document.getElementById('quality-result');
    
    const qualityBadgeClass = quality.class === 'Good_quality_coal' ? 
        'quality-good result-badge' : 'quality-bad result-badge';
    
    const qualityLabel = quality.class === 'Good_quality_coal' ? 
        'فحم عالي الجودة' : 'فحم معيب';
    
    qualityElement.innerHTML = `
        <div class="d-flex align-items-center mb-3">
            <span class="${qualityBadgeClass}">
                <i class="fas ${quality.class === 'Good_quality_coal' ? 'fa-check-circle' : 'fa-exclamation-triangle'} me-2"></i>
                ${qualityLabel}
            </span>
            <span class="ms-3 fs-5 fw-bold">${(quality.confidence * 100).toFixed(2)}%</span>
        </div>
        <div class="quality-probabilities">
            ${Object.entries(quality.probabilities).map(([className, prob]) => `
                <div class="probability-item mb-2">
                    <div class="d-flex justify-content-between">
                        <span>${getArabicLabel(className)}</span>
                        <span>${(prob * 100).toFixed(2)}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill ${className === 'Good_quality_coal' ? 'quality-good' : 'quality-bad'}" 
                             style="width: ${prob * 100}%">
                            ${(prob * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    // Display defect results if coal is defective
    const defectResults = document.getElementById('defect-results');
    const defectChartContainer = document.getElementById('defect-chart-container');
    
    if (quality.class === 'Defect_coal' && data.defect_prediction) {
        const defect = data.defect_prediction;
        const defectElement = document.getElementById('defect-result');
        
        const defectLabel = defect.class === 'Cracks_and_fractures' ? 
            'شقوق وكسور' : 'تشوه السطح';
        
        defectElement.innerHTML = `
            <div class="d-flex align-items-center mb-3">
                <span class="defect-${defect.class === 'Cracks_and_fractures' ? 'crack' : 'surface'} result-badge">
                    <i class="fas ${defect.class === 'Cracks_and_fractures' ? 'fa-bolt' : 'fa-mountain'} me-2"></i>
                    ${defectLabel}
                </span>
                <span class="ms-3 fs-5 fw-bold">${(defect.confidence * 100).toFixed(2)}%</span>
            </div>
            <div class="defect-probabilities">
                ${Object.entries(defect.probabilities).map(([className, prob]) => `
                    <div class="probability-item mb-2">
                        <div class="d-flex justify-content-between">
                            <span>${getArabicLabel(className)}</span>
                            <span>${(prob * 100).toFixed(2)}%</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill ${className === 'Cracks_and_fractures' ? 'defect-crack' : 'defect-surface'}" 
                                 style="width: ${prob * 100}%">
                                ${(prob * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        defectResults.classList.remove('d-none');
        defectChartContainer.classList.remove('d-none');
        
        // Create defect chart
        createDefectChart(defect.probabilities);
    } else {
        defectResults.classList.add('d-none');
        defectChartContainer.classList.add('d-none');
    }
    
    // Create quality chart
    createQualityChart(quality.probabilities);
    
    // Show results with animation
    resultsSection.classList.remove('d-none');
    setTimeout(() => {
        resultsSection.classList.add('fade-in-up');
    }, 100);
}

// Create quality probability chart
function createQualityChart(probabilities) {
    const ctx = document.getElementById('quality-chart');
    
    if (!ctx) {
        console.error('Quality chart canvas not found');
        return;
    }
    
    if (qualityChart) {
        qualityChart.destroy();
    }
    
    const labels = Object.keys(probabilities).map(getArabicLabel);
    const data = Object.values(probabilities);
    const backgroundColors = [
        'rgba(220, 53, 69, 0.8)',  // Defect_coal - red
        'rgba(40, 167, 69, 0.8)'   // Good_quality_coal - green
    ];
    
    qualityChart = new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderColor: ['rgba(220, 53, 69, 1)', 'rgba(40, 167, 69, 1)'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    rtl: true
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${(context.raw * 100).toFixed(2)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Create defect probability chart
function createDefectChart(probabilities) {
    const ctx = document.getElementById('defect-chart');
    
    if (!ctx) {
        console.error('Defect chart canvas not found');
        return;
    }
    
    if (defectChart) {
        defectChart.destroy();
    }
    
    const labels = Object.keys(probabilities).map(getArabicLabel);
    const data = Object.values(probabilities);
    const backgroundColors = [
        'rgba(255, 107, 107, 0.8)',  // Cracks_and_fractures
        'rgba(72, 52, 212, 0.8)'     // Surface_deformation
    ];
    
    defectChart = new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderColor: ['rgba(255, 107, 107, 1)', 'rgba(72, 52, 212, 1)'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    rtl: true
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${(context.raw * 100).toFixed(2)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Get Arabic labels
function getArabicLabel(className) {
    const labels = {
        'Good_quality_coal': 'فحم عالي الجودة',
        'Defect_coal': 'فحم معيب',
        'Cracks_and_fractures': 'شقوق وكسور',
        'Surface_deformation': 'تشوه السطح'
    };
    return labels[className] || className;
}

// Check model status
function checkModelStatus() {
    fetch('/model_status')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error' || data.status === 'warning') {
                modelStatusText.textContent = data.message;
                modelStatus.classList.remove('d-none');
                
                if (data.status === 'error') {
                    modelStatus.classList.remove('alert-warning');
                    modelStatus.classList.add('alert-danger');
                }
            }
        })
        .catch(error => {
            console.error('Model status check failed:', error);
        });
}

// Utility functions
function showLoading() {
    loadingSpinner.classList.remove('d-none');
}

function hideLoading() {
    loadingSpinner.classList.add('d-none');
}

function hideResults() {
    resultsSection.classList.add('d-none');
    resultsSection.classList.remove('fade-in-up');
}

function showError(message) {
    errorMessage.textContent = message;
    errorAlert.classList.remove('d-none');
    
    // Scroll to error
    errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    errorAlert.classList.add('d-none');
}