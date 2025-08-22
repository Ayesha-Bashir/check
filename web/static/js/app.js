// JavaScript for Movie Recommendation System Web Interface

let currentSessionId = null;
let currentAnalysisData = null;

// Step tracking for one-time execution
let stepsCompleted = {
    dataset: false,
    analysis: false,
    visualization: false,
    download: false
};

let currentStep = 1;
const totalSteps = 4;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    setupFormValidation();
    initializeStepControl();
});

/**
 * Set up all event listeners for the application
 */
function setupEventListeners() {
    // Dataset type change handler
    document.getElementById('dataset_type').addEventListener('change', function() {
        toggleDatasetParameters(this.value);
    });

    // Algorithm type change handler
    document.getElementById('algorithm_type').addEventListener('change', function() {
        toggleAlgorithmParameters(this.value);
    });

    // Form submission handlers
    document.getElementById('datasetForm').addEventListener('submit', handleDatasetGeneration);
    document.getElementById('analysisForm').addEventListener('submit', handleAnalysisRun);

    // Initialize with realistic parameters visible
    toggleDatasetParameters('realistic');
    toggleAlgorithmParameters('original');
}

/**
 * Initialize step control system
 */
function initializeStepControl() {
    // Initially disable all steps except the first one
    updateStepAvailability();
    
    // Add visual indicators for step completion
    addStepIndicators();
}

/**
 * Set up form validation
 */
function setupFormValidation() {
    // Add real-time validation for numeric inputs
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            validateNumericInput(this);
        });
    });
}

/**
 * Toggle dataset parameter visibility based on selected type
 * @param {string} datasetType - The selected dataset type
 */
function toggleDatasetParameters(datasetType) {
    const realisticParams = document.getElementById('realistic_params');
    const clusteredParams = document.getElementById('clustered_params');
    const simpleParams = document.getElementById('simple_params');

    // Hide all parameter sections
    realisticParams.style.display = 'none';
    clusteredParams.style.display = 'none';
    simpleParams.style.display = 'none';

    // Show relevant parameters
    switch(datasetType) {
        case 'realistic':
            realisticParams.style.display = 'block';
            break;
        case 'clustered':
            clusteredParams.style.display = 'block';
            break;
        case 'simple':
            simpleParams.style.display = 'block';
            break;
    }

    // Add fade-in animation
    const activeSection = document.getElementById(`${datasetType}_params`);
    if (activeSection) {
        activeSection.classList.add('fade-in');
    }
}

/**
 * Toggle algorithm parameter visibility based on selected type
 * @param {string} algorithmType - The selected algorithm type
 */
function toggleAlgorithmParameters(algorithmType) {
    const originalParams = document.getElementById('original_params');
    const extendedParams = document.getElementById('extended_params');
    const kNeighborsDiv = document.getElementById('k_neighbors_div');

    if (algorithmType === 'original') {
        originalParams.style.display = 'block';
        extendedParams.style.display = 'none';
        kNeighborsDiv.style.display = 'block';
    } else {
        originalParams.style.display = 'none';
        extendedParams.style.display = 'block';
        kNeighborsDiv.style.display = 'none';
    }
}

/**
 * Validate numeric inputs
 * @param {HTMLElement} input - The input element to validate
 */
function validateNumericInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);

    input.classList.remove('is-invalid', 'is-valid');

    if (isNaN(value) || value < min || value > max) {
        input.classList.add('is-invalid');
    } else {
        input.classList.add('is-valid');
    }
}

/**
 * Update step availability based on current progress
 */
function updateStepAvailability() {
    // Step 1: Dataset generation - always available if not completed
    const datasetBtn = document.getElementById('datasetBtn');
    datasetBtn.disabled = stepsCompleted.dataset;
    
    // Step 2: Analysis - available only after dataset is generated
    const analysisBtn = document.getElementById('analysisBtn');
    analysisBtn.disabled = !stepsCompleted.dataset || stepsCompleted.analysis;
    
    // Step 3: Visualization - available only after analysis
    const analysisVizBtn = document.getElementById('analysisVizBtn');
    const recVizBtn = document.getElementById('recVizBtn');
    const visualizationDisabled = !stepsCompleted.analysis || stepsCompleted.visualization;
    analysisVizBtn.disabled = visualizationDisabled;
    recVizBtn.disabled = visualizationDisabled;
    
    // Step 4: Download - available only after analysis
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.disabled = !stepsCompleted.analysis || stepsCompleted.download;
    }
}

/**
 * Add visual indicators for step completion
 */
function addStepIndicators() {
    const stepHeaders = [
        { id: 'step1Header', selector: '.card:nth-child(1) .card-header h3' },
        { id: 'step2Header', selector: '.card:nth-child(2) .card-header h3' },
        { id: 'step3Header', selector: '.card:nth-child(3) .card-header h3' },
        { id: 'step4Header', selector: '.card:nth-child(4) .card-header h3' }
    ];
    
    stepHeaders.forEach((header, index) => {
        const element = document.querySelector(header.selector);
        if (element) {
            element.id = header.id;
            const stepKey = ['dataset', 'analysis', 'visualization', 'download'][index];
            updateStepIndicator(header.id, stepsCompleted[stepKey]);
        }
    });
}

/**
 * Update visual indicator for a specific step
 * @param {string} headerId - ID of the step header
 * @param {boolean} completed - Whether the step is completed
 */
function updateStepIndicator(headerId, completed) {
    const header = document.getElementById(headerId);
    if (header) {
        // Remove existing indicators
        const existingIcon = header.querySelector('.step-status-icon');
        if (existingIcon) {
            existingIcon.remove();
        }
        
        // Get the card element
        const card = header.closest('.card');
        
        if (completed) {
            // Add completed indicator
            const completedIcon = document.createElement('i');
            completedIcon.className = 'fas fa-check-circle step-status-icon text-success ms-2';
            completedIcon.title = 'Completed';
            header.appendChild(completedIcon);
            
            // Add completed styling to card
            if (card) {
                card.classList.add('step-completed');
            }
        } else {
            // Remove completed styling from card
            if (card) {
                card.classList.remove('step-completed');
            }
        }
    }
}

/**
 * Mark a step as completed and update UI
 * @param {string} stepName - Name of the completed step
 */
function completeStep(stepName) {
    stepsCompleted[stepName] = true;
    
    // Update visual indicators
    const headerIds = ['step1Header', 'step2Header', 'step3Header', 'step4Header'];
    const stepKeys = ['dataset', 'analysis', 'visualization', 'download'];
    const index = stepKeys.indexOf(stepName);
    
    if (index !== -1) {
        updateStepIndicator(headerIds[index], true);
    }
    
    // Update button availability
    updateStepAvailability();
    
    // Check if all steps are completed
    if (Object.values(stepsCompleted).every(completed => completed)) {
        showRestartSection();
    }
}

/**
 * Show the restart section after all steps are completed
 */
function showRestartSection() {
    hideElement('restartPlaceholder');
    showElement('restartControls');
    document.getElementById('restartControls').classList.add('fade-in');
}

/**
 * Restart the entire process
 */
function restartProcess() {
    // Reset step tracking
    stepsCompleted = {
        dataset: false,
        analysis: false,
        visualization: false,
        download: false
    };
    
    // Reset session data
    currentSessionId = null;
    currentAnalysisData = null;
    
    // Reset UI elements
    hideElement('datasetResult');
    hideElement('analysisResult');
    hideElement('analysisForm');
    hideElement('visualizationControls');
    hideElement('downloadControls');
    hideElement('downloadResult');
    hideElement('restartControls');
    
    showElement('analysisPlaceholder');
    showElement('visualizationPlaceholder');
    showElement('downloadPlaceholder');
    showElement('restartPlaceholder');
    
    // Clear visualization
    hideElement('plotImage');
    
    // Reset forms
    document.getElementById('datasetForm').reset();
    document.getElementById('analysisForm').reset();
    
    // Reset step indicators
    ['step1Header', 'step2Header', 'step3Header', 'step4Header'].forEach(headerId => {
        updateStepIndicator(headerId, false);
    });
    
    // Update button availability
    updateStepAvailability();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    showSuccess('Process Restarted', 'You can now begin a new analysis.');
}

// Make restartProcess available globally for HTML onclick
window.restartProcess = restartProcess;

/**
 * Handle dataset generation form submission
 * @param {Event} event - The form submission event
 */
async function handleDatasetGeneration(event) {
    event.preventDefault();
    
    // Prevent re-execution if already completed
    if (stepsCompleted.dataset) {
        showError('Step Already Completed', 'Dataset has already been generated. Use restart to begin a new analysis.');
        return;
    }
    
    const form = event.target;
    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    
    // Show loading state
    setButtonLoading(submitBtn, true);
    hideElement('datasetResult');
    
    try {
        const response = await fetch('/generate_dataset', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentSessionId = result.session_id;
            showDatasetSuccess(result);
            enableAnalysisSection();
            completeStep('dataset');
        } else {
            showError('Dataset Generation Failed', result.error);
        }
    } catch (error) {
        showError('Network Error', 'Failed to connect to server. Please try again.');
        console.error('Dataset generation error:', error);
    } finally {
        setButtonLoading(submitBtn, false);
    }
}

/**
 * Show dataset generation success
 * @param {Object} result - The success result from server
 */
function showDatasetSuccess(result) {
    const statsDiv = document.getElementById('datasetStats');
    const stats = result.statistics;
    
    statsDiv.innerHTML = `
        <strong>Dataset Statistics:</strong>
        <ul class="mb-0 mt-2">
            <li>Total Users: <span class="badge bg-primary">${stats.total_users}</span></li>
            <li>Total Movies: <span class="badge bg-info">${stats.total_movies}</span></li>
            <li>Total Ratings: <span class="badge bg-success">${stats.total_ratings}</span></li>
            <li>Sparsity: <span class="badge bg-warning text-dark">${(stats.actual_sparsity * 100).toFixed(1)}%</span></li>
            <li>Avg Ratings/User: <span class="badge bg-secondary">${stats.avg_ratings_per_user.toFixed(1)}</span></li>
        </ul>
    `;
    
    showElement('datasetResult');
    
    // Update target user max value
    const targetUserInput = document.getElementById('target_user');
    targetUserInput.max = stats.total_users - 1;
    targetUserInput.value = Math.min(targetUserInput.value, stats.total_users - 1);
}

/**
 * Enable the analysis section after dataset generation
 */
function enableAnalysisSection() {
    hideElement('analysisPlaceholder');
    showElement('analysisForm');
    document.getElementById('analysisForm').classList.add('fade-in');
}

/**
 * Handle analysis form submission
 * @param {Event} event - The form submission event
 */
async function handleAnalysisRun(event) {
    event.preventDefault();
    
    // Prevent re-execution if already completed
    if (stepsCompleted.analysis) {
        showError('Step Already Completed', 'Analysis has already been run. Use restart to begin a new analysis.');
        return;
    }
    
    const form = event.target;
    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    
    // Convert form data to JSON
    const analysisData = {
        session_id: currentSessionId,
        algorithm_type: formData.get('algorithm_type'),
        target_user: parseInt(formData.get('target_user')),
        n_recommendations: parseInt(formData.get('n_recommendations'))
    };
    
    // Add algorithm-specific parameters
    if (analysisData.algorithm_type === 'original') {
        analysisData.similarity_method = formData.get('similarity_method');
        analysisData.recommendation_method = formData.get('recommendation_method');
        analysisData.k_neighbors = parseInt(formData.get('k_neighbors'));
    } else {
        analysisData.n_factors = parseInt(formData.get('n_factors'));
    }
    
    // Show loading state
    setButtonLoading(submitBtn, true);
    hideElement('analysisResult');
    
    try {
        const response = await fetch('/run_analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(analysisData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentAnalysisData = analysisData;
            showAnalysisSuccess(result);
            enableVisualizationSection();
            enableDownloadSection();
            completeStep('analysis');
        } else {
            showError('Analysis Failed', result.error);
        }
    } catch (error) {
        showError('Network Error', 'Failed to connect to server. Please try again.');
        console.error('Analysis error:', error);
    } finally {
        setButtonLoading(submitBtn, false);
    }
}

/**
 * Show analysis success results
 * @param {Object} result - The success result from server
 */
function showAnalysisSuccess(result) {
    const recList = document.getElementById('recommendationsList');
    
    let html = `
        <strong>Recommendations for User ${result.user_id}</strong><br>
        <small class="text-muted">Algorithm: ${result.method_description || result.method}</small>
        <div class="mt-2">
    `;
    
    result.recommendations.forEach((rec, index) => {
        html += `
            <div class="recommendation-item">
                <span class="fw-bold">#${index + 1} Movie ${rec.movie_id}</span>
                <span class="rating-badge float-end">★ ${rec.predicted_rating}</span>
            </div>
        `;
    });
    
    html += '</div>';
    recList.innerHTML = html;
    
    showElement('analysisResult');
}

/**
 * Enable visualization section after analysis
 */
function enableVisualizationSection() {
    hideElement('visualizationPlaceholder');
    showElement('visualizationControls');
    document.getElementById('visualizationControls').classList.add('fade-in');
    
    // Update button availability
    updateStepAvailability();
}

/**
 * Enable download section after analysis
 */
function enableDownloadSection() {
    hideElement('downloadPlaceholder');
    showElement('downloadControls');
    document.getElementById('downloadControls').classList.add('fade-in');
    
    // Update button availability
    updateStepAvailability();
}

/**
 * Generate visualization plots
 * @param {string} plotType - Type of plot to generate ('analysis' or 'recommendations')
 */
async function generateVisualization(plotType) {
    // Prevent re-execution if already completed
    if (stepsCompleted.visualization) {
        showError('Step Already Completed', 'Visualizations have already been generated. Use restart to begin a new analysis.');
        return;
    }
    
    const loadingSpinner = document.getElementById('loadingSpinner');
    const plotImage = document.getElementById('plotImage');
    
    // Show loading
    showElement('loadingSpinner');
    hideElement('plotImage');
    
    const requestData = {
        session_id: currentSessionId,
        plot_type: plotType
    };
    
    // Add additional data for recommendations plot
    if (plotType === 'recommendations' && currentAnalysisData) {
        requestData.target_user = currentAnalysisData.target_user;
        requestData.method = currentAnalysisData.recommendation_method;
    }
    
    try {
        const response = await fetch('/generate_visualization', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            plotImage.src = `data:image/png;base64,${result.image}`;
            hideElement('loadingSpinner');
            showElement('plotImage');
            plotImage.classList.add('fade-in');
            
            // Mark visualization step as completed after first successful generation
            completeStep('visualization');
        } else {
            showError('Visualization Failed', result.error);
            hideElement('loadingSpinner');
        }
    } catch (error) {
        showError('Network Error', 'Failed to generate visualization.');
        console.error('Visualization error:', error);
        hideElement('loadingSpinner');
    }
}

/**
 * Download analysis results package
 */
async function downloadResults() {
    if (!currentSessionId) {
        showError('No Data', 'Please generate a dataset and run analysis first.');
        return;
    }
    
    // Prevent re-execution if already completed
    if (stepsCompleted.download) {
        showError('Step Already Completed', 'Results have already been downloaded. Use restart to begin a new analysis.');
        return;
    }
    
    try {
        const response = await fetch('/download_results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: currentSessionId
            })
        });
        
        if (response.ok) {
            // Create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `movie_recommendations_${currentSessionId}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            // Show download result section
            showElement('downloadResult');
            showSuccess('Download Complete', 'Analysis package downloaded successfully!');
            
            // Mark download step as completed
            completeStep('download');
        } else {
            const result = await response.json();
            showError('Download Failed', result.error || 'Failed to create download package.');
        }
    } catch (error) {
        showError('Network Error', 'Failed to download results.');
        console.error('Download error:', error);
    }
}

/**
 * Utility Functions
 */

/**
 * Set button loading state
 * @param {HTMLElement} button - The button element
 * @param {boolean} loading - Whether to show loading state
 */
function setButtonLoading(button, loading) {
    const originalText = button.getAttribute('data-original-text') || button.innerHTML;
    
    if (loading) {
        button.setAttribute('data-original-text', originalText);
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        button.disabled = true;
    } else {
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

/**
 * Show element
 * @param {string} elementId - ID of element to show
 */
function showElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'block';
    }
}

/**
 * Hide element
 * @param {string} elementId - ID of element to hide
 */
function hideElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'none';
    }
}

/**
 * Show error message
 * @param {string} title - Error title
 * @param {string} message - Error message
 */
function showError(title, message) {
    // Create error alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.innerHTML = `
        <strong><i class="fas fa-exclamation-triangle me-2"></i>${title}</strong>
        <br>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

/**
 * Show success message
 * @param {string} title - Success title
 * @param {string} message - Success message
 */
function showSuccess(title, message) {
    // Create success alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show';
    alertDiv.innerHTML = `
        <strong><i class="fas fa-check-circle me-2"></i>${title}</strong>
        <br>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 3000);
}
