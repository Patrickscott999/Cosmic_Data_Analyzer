document.addEventListener('DOMContentLoaded', function() {
    // Initialize elements
    const uploadForm = document.getElementById('upload-form');
    const fileUpload = document.getElementById('file-upload');
    const uploadSection = document.getElementById('upload-section');
    const analysisSection = document.getElementById('analysis-section');
    const datasetInfo = document.getElementById('dataset-info');
    const datasetName = document.getElementById('dataset-name');
    const datasetRows = document.getElementById('dataset-rows');
    const datasetCols = document.getElementById('dataset-cols');
    const cleanDataBtn = document.getElementById('clean-data-btn');
    const mlForm = document.getElementById('ml-form');
    const mineForm = document.getElementById('mine-form');
    const visualizeForm = document.getElementById('visualize-form');
    const forecastForm = document.getElementById('forecast-form');
    const compareForm = document.getElementById('compare-form');
    const generateInsightsBtn = document.getElementById('generate-insights-btn');
    const chatForm = document.getElementById('chat-form');
    const nlqueryForm = document.getElementById('nlquery-form');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    // Initialize Feather Icons
    feather.replace();
    
    // Add fade-in animation to main content
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.style.opacity = '0';
        mainContent.style.transform = 'translateY(20px)';
        mainContent.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        
        // Slight delay to ensure smooth animation
        setTimeout(() => {
            mainContent.style.opacity = '1';
            mainContent.style.transform = 'translateY(0)';
        }, 200);
    }
    
    // Animate all cards with staggered delay
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        
        // Staggered delay for each card
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 200 + (index * 150)); // 150ms delay between each card
    });
    
    // Initialize warp transition effect
    const warpTransition = document.querySelector('.warp-transition');
    
    // Function to trigger warp effect when navigating
    function triggerWarpEffect() {
        if (warpTransition) {
            warpTransition.classList.add('active');
            setTimeout(() => {
                warpTransition.classList.remove('active');
            }, 600);
        }
    }
    
    // Trigger warp effect for navigation links
    document.querySelectorAll('a:not([target="_blank"])').forEach(link => {
        link.addEventListener('click', (e) => {
            // Only trigger for navigation links (not # links, not external links)
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !href.startsWith('http') && !href.startsWith('javascript:')) {
                e.preventDefault();
                triggerWarpEffect();
                setTimeout(() => {
                    window.location.href = href;
                }, 500);
            }
        });
    });
    
    // Add animated effect to feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(40px)';
        card.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 400 + (index * 200)); // 200ms delay between feature cards
    });
    
    // Toast notification
    const toastElement = document.getElementById('toast-notification');
    const toast = new bootstrap.Toast(toastElement);
    const toastTitle = document.getElementById('toast-title');
    const toastMessage = document.getElementById('toast-message');
    
    // Show a notification with animation
    function showNotification(title, message, type = 'info') {
        toastTitle.textContent = title;
        toastMessage.textContent = message;
        
        // Get the toast icon and set it based on notification type
        const toastIcon = document.getElementById('toast-icon');
        const iconContainer = document.querySelector('.cosmic-toast-icon');
        
        if (toastIcon && iconContainer) {
            // Set icon and color based on notification type
            if (type === 'success') {
                toastIcon.setAttribute('data-feather', 'check-circle');
                iconContainer.style.background = 'linear-gradient(135deg, #10B981, #059669)';
            } else if (type === 'error') {
                toastIcon.setAttribute('data-feather', 'alert-circle');
                iconContainer.style.background = 'linear-gradient(135deg, #EF4444, #B91C1C)';
            } else if (type === 'warning') {
                toastIcon.setAttribute('data-feather', 'alert-triangle');
                iconContainer.style.background = 'linear-gradient(135deg, #F59E0B, #B45309)';
            } else {
                toastIcon.setAttribute('data-feather', 'info');
                iconContainer.style.background = 'linear-gradient(135deg, var(--theme-primary), var(--theme-info))';
            }
            
            // Re-initialize feather icons
            feather.replace();
        } else {
            toastElement.classList.add('bg-info', 'text-white');
            const icon = toastElement.querySelector('.toast-header i');
            if (icon) {
                icon.setAttribute('data-feather', 'info');
                feather.replace();
            }
        }
        
        // Add shake animation for errors
        if (type === 'error') {
            toastElement.classList.add('shake-animation');
            setTimeout(() => {
                toastElement.classList.remove('shake-animation');
            }, 500);
        }
        
        // Add animation effect to toast message text
        if (toastMessage) {
            toastMessage.style.opacity = '0';
            toastMessage.style.transform = 'translateX(-10px)';
            toastMessage.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            
            setTimeout(() => {
                toastMessage.style.opacity = '1';
                toastMessage.style.transform = 'translateX(0)';
            }, 100);
        }
        
        toast.show();
    }
    
    // Add some CSS for toast animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        .shake-animation {
            animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .pulse-animation {
            animation: pulse 0.5s ease;
        }
        
        .viz-type-option:hover {
            border-color: var(--theme-primary) !important;
            background-color: rgba(99, 102, 241, 0.05) !important;
        }
        
        .viz-type-option input:checked + label {
            border-color: var(--theme-primary) !important;
            background-color: rgba(99, 102, 241, 0.1) !important;
        }
    `;
    document.head.appendChild(style);
    
    // Show loading spinner with animation
    function showLoading() {
        loadingOverlay.style.display = 'flex';
        loadingOverlay.style.opacity = '0';
        setTimeout(() => {
            loadingOverlay.style.opacity = '1';
        }, 10);
    }
    
    // Hide loading spinner with animation
    function hideLoading() {
        loadingOverlay.style.opacity = '0';
        setTimeout(() => {
            loadingOverlay.style.display = 'none';
        }, 300);
    }
    
    // Handle file upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileUpload.files[0]) {
            showNotification('Error', 'Please select a file to upload', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileUpload.files[0]);
        
        showLoading();
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            // Store the dataset ID for future requests
            currentDatasetId = data.dataset_id;
            // Store in session storage for persistence between page refreshes
            sessionStorage.setItem('current_dataset_id', currentDatasetId);
            console.log("Dataset ID stored:", currentDatasetId);
            
            // Smooth transition between upload and analysis sections
            uploadSection.style.opacity = '1';
            setTimeout(() => {
                uploadSection.style.opacity = '0';
                uploadSection.style.transform = 'translateY(-20px)';
                
                setTimeout(() => {
                    // Hide upload and show analysis section
                    uploadSection.style.display = 'none';
                    
                    // Initialize and show analysis section
                    analysisSection.style.opacity = '0';
                    analysisSection.style.transform = 'translateY(20px)';
                    analysisSection.style.display = 'block';
                    
                    // Show file info in the header
                    datasetInfo.style.display = 'flex !important';
                    datasetInfo.style.removeProperty('display');
                    datasetName.textContent = data.filename;
                    datasetRows.textContent = data.rows;
                    datasetCols.textContent = data.columns.length;
                    
                    // Animate in the analysis section
                    setTimeout(() => {
                        analysisSection.style.opacity = '1';
                        analysisSection.style.transform = 'translateY(0)';
                        
                        // Reinitialize Feather icons for new content
                        feather.replace();
                        
                        // Show export options and dispatch dataset loaded event
                        showDataExportOptions();
                        document.dispatchEvent(new CustomEvent('datasetLoaded', {
                            detail: { datasetId: currentDatasetId }
                        }));
                    }, 50);
                }, 300);
            }, 50);
            
            // Populate column selectors
            populateColumnSelectors(data.column_names, data.dtypes);
            
            // Load data preview
            loadDataPreview();
            
            showNotification('Success', 'File uploaded successfully', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to upload file', 'error');
        });
    });
    
    // Populate column selectors for various forms
    function populateColumnSelectors(columns, dtypes) {
        // Target column for ML
        const targetColumn = document.getElementById('target-column');
        targetColumn.innerHTML = '<option value="">Select a column</option>';
        
        // X and Y axis for visualization
        const xAxis = document.getElementById('x-axis');
        const yAxis = document.getElementById('y-axis');
        xAxis.innerHTML = '<option value="">Select a column</option>';
        yAxis.innerHTML = '<option value="">Select a column</option>';
        
        // Columns for mining
        const miningColumns = document.getElementById('mining-columns');
        miningColumns.innerHTML = '';
        
        // Columns for heatmap
        const heatmapColumns = document.getElementById('heatmap-columns');
        heatmapColumns.innerHTML = '';
        
        // Time series forecasting form fields
        const dateColumn = document.getElementById('date-column');
        const forecastColumn = document.getElementById('target-column');
        
        if (dateColumn) {
            dateColumn.innerHTML = '<option value="" selected disabled>Select date column</option>';
        }
        
        if (forecastColumn) {
            forecastColumn.innerHTML = '<option value="" selected disabled>Select target column</option>';
        }
        
        // Update dataset comparison dropdowns
        updateDatasetDropdowns();
        
        function updateDatasetDropdowns() {
            fetch('/datasets')
                .then(response => response.json())
                .then(data => {
                    const dataset1 = document.getElementById('dataset1');
                    const dataset2 = document.getElementById('dataset2');
                    
                    if (dataset1 && dataset2 && data.datasets) {
                        dataset1.innerHTML = '<option value="" selected disabled>Select a dataset</option>';
                        dataset2.innerHTML = '<option value="" selected disabled>Select a dataset</option>';
                        
                        Object.keys(data.datasets).forEach(id => {
                            const dataset = data.datasets[id];
                            
                            const option1 = document.createElement('option');
                            option1.value = id;
                            option1.textContent = dataset.name || `Dataset ${id}`;
                            dataset1.appendChild(option1);
                            
                            const option2 = document.createElement('option');
                            option2.value = id;
                            option2.textContent = dataset.name || `Dataset ${id}`;
                            dataset2.appendChild(option2);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error fetching datasets:', error);
                });
        }
        
        // Populate selectors
        columns.forEach(column => {
            // Add to target column selector
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            targetColumn.appendChild(option.cloneNode(true));
            
            // Add to X and Y axis selectors
            xAxis.appendChild(option.cloneNode(true));
            yAxis.appendChild(option.cloneNode(true));
            
            // Add to mining columns as checkboxes
            const miningCheck = document.createElement('div');
            miningCheck.className = 'column-checkbox form-check';
            miningCheck.innerHTML = `
                <input class="form-check-input" type="checkbox" value="${column}" id="mine-${column}" name="mining-columns" checked>
                <label class="form-check-label" for="mine-${column}">${column}</label>
            `;
            miningColumns.appendChild(miningCheck);
            
            // Add numeric columns to heatmap selector
            if (dtypes[column].includes('int') || dtypes[column].includes('float')) {
                const heatmapCheck = document.createElement('div');
                heatmapCheck.className = 'column-checkbox form-check';
                heatmapCheck.innerHTML = `
                    <input class="form-check-input" type="checkbox" value="${column}" id="heatmap-${column}" name="heatmap-columns" checked>
                    <label class="form-check-label" for="heatmap-${column}">${column}</label>
                `;
                heatmapColumns.appendChild(heatmapCheck);
            }
            
            // Add to date column for forecasting
            if (dateColumn) {
                const dateOption = document.createElement('option');
                dateOption.value = column;
                dateOption.textContent = column;
                dateColumn.appendChild(dateOption);
            }
            
            // Add to target column for forecasting (numeric columns only)
            if (forecastColumn && dtypes && dtypes[column] && (dtypes[column].includes('int') || dtypes[column].includes('float'))) {
                const forecastOption = document.createElement('option');
                forecastOption.value = column;
                forecastOption.textContent = column;
                forecastColumn.appendChild(forecastOption);
            }
        });
    }
    
    // Store the current dataset ID
    let currentDatasetId = null;
    
    // Load data preview
    function loadDataPreview() {
        showLoading();
        
        // Use the stored dataset ID if available
        const url = currentDatasetId ? `/data-preview?dataset_id=${currentDatasetId}` : '/data-preview';
        
        fetch(url)
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            // Populate the table
            const tableHeader = document.getElementById('data-preview-header');
            const tableBody = document.getElementById('data-preview-body');
            
            // Clear existing content
            tableHeader.innerHTML = '';
            tableBody.innerHTML = '';
            
            // Add column headers
            data.columns.forEach(column => {
                const th = document.createElement('th');
                th.textContent = column;
                tableHeader.appendChild(th);
            });
            
            // Add data rows
            data.preview.forEach(row => {
                const tr = document.createElement('tr');
                
                data.columns.forEach(column => {
                    const td = document.createElement('td');
                    td.textContent = row[column] !== null ? row[column] : 'null';
                    tr.appendChild(td);
                });
                
                tableBody.appendChild(tr);
            });
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to load data preview', 'error');
        });
    }
    
    // Function to show advanced cleaning options
    function showCleaningOptions() {
        // Create modal for cleaning options
        const modalHTML = `
        <div class="modal fade" id="cleaningOptionsModal" tabindex="-1" aria-labelledby="cleaningOptionsModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="cleaningOptionsModalLabel">Advanced Data Cleaning Options</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="dropIdColumns" checked>
                                    <label class="form-check-label" for="dropIdColumns">Auto-detect and drop ID-like columns</label>
                                    <small class="form-text text-muted d-block">Removes columns that look like IDs (high uniqueness)</small>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="handleOutliers" checked>
                                    <label class="form-check-label" for="handleOutliers">Handle outliers</label>
                                    <small class="form-text text-muted d-block">Caps extreme values using IQR method</small>
                                </div>
                            </div>
                        </div>
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="normalizeNumeric">
                                    <label class="form-check-label" for="normalizeNumeric">Normalize numeric columns</label>
                                    <small class="form-text text-muted d-block">Scales numeric values to have mean=0, std=1</small>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="textCleaning" checked>
                                    <label class="form-check-label" for="textCleaning">Clean text columns</label>
                                    <small class="form-text text-muted d-block">Lowercases, strips spaces, removes special chars</small>
                                </div>
                            </div>
                        </div>
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <label for="nullThreshold" class="form-label">Drop columns with null values above:</label>
                                <div class="input-group">
                                    <input type="range" class="form-range" id="nullThreshold" min="0" max="1" step="0.05" value="0.5">
                                    <span id="nullThresholdValue" class="ms-2">50%</span>
                                </div>
                                <small class="form-text text-muted d-block">Columns with more nulls than this % will be dropped</small>
                            </div>
                            <div class="col-md-6">
                                <label for="duplicateHandling" class="form-label">Handle duplicate rows:</label>
                                <select class="form-select" id="duplicateHandling">
                                    <option value="drop" selected>Drop duplicates</option>
                                    <option value="mark">Mark duplicates</option>
                                    <option value="none">Keep all rows</option>
                                </select>
                                <small class="form-text text-muted d-block">How to handle duplicate rows in the dataset</small>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="applyCleaningBtn">Apply Cleaning</button>
                    </div>
                </div>
            </div>
        </div>
        `;
        
        // Append modal HTML to body if it doesn't exist
        if (!document.getElementById('cleaningOptionsModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHTML);
            
            // Handle null threshold slider
            const nullThreshold = document.getElementById('nullThreshold');
            const nullThresholdValue = document.getElementById('nullThresholdValue');
            
            nullThreshold.addEventListener('input', function() {
                nullThresholdValue.textContent = `${Math.round(this.value * 100)}%`;
            });
            
            // Handle apply button
            document.getElementById('applyCleaningBtn').addEventListener('click', function() {
                // Get all options
                const options = {
                    drop_id_columns: document.getElementById('dropIdColumns').checked,
                    handle_outliers: document.getElementById('handleOutliers').checked,
                    normalize_numeric: document.getElementById('normalizeNumeric').checked,
                    text_cleaning: document.getElementById('textCleaning').checked,
                    drop_high_null_threshold: parseFloat(document.getElementById('nullThreshold').value),
                    duplicate_handling: document.getElementById('duplicateHandling').value
                };
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('cleaningOptionsModal'));
                modal.hide();
                
                // Call clean function with options
                cleanDataWithOptions(options);
            });
        }
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('cleaningOptionsModal'));
        modal.show();
    }
    
    // Function to clean data with options
    function cleanDataWithOptions(options) {
        showLoading();
        
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            hideLoading();
            return;
        }
        
        fetch('/clean', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: currentDatasetId,
                options: options
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            // Update dataset info
            datasetRows.textContent = data.rows_after;
            datasetCols.textContent = data.columns.length;
            
            // Update column selectors
            populateColumnSelectors(data.columns, data.dtypes);
            
            // Refresh data preview
            loadDataPreview();
            
            // Show detailed cleaning results
            const cleanResult = document.getElementById('clean-result');
            
            // Build lists of operations performed
            let operationLists = '';
            
            if (data.cleaning_details.dropped_columns && data.cleaning_details.dropped_columns.length > 0) {
                operationLists += `<h6>Dropped Columns (${data.cleaning_details.dropped_columns.length}):</h6>
                <ul class="small">
                    ${data.cleaning_details.dropped_columns.map(col => `<li>${col}</li>`).join('')}
                </ul>`;
            }
            
            if (data.cleaning_details.converted_columns && data.cleaning_details.converted_columns.length > 0) {
                operationLists += `<h6>Converted Columns (${data.cleaning_details.converted_columns.length}):</h6>
                <ul class="small">
                    ${data.cleaning_details.converted_columns.map(col => `<li>${col}</li>`).join('')}
                </ul>`;
            }
            
            if (data.cleaning_details.encoded_columns && data.cleaning_details.encoded_columns.length > 0) {
                operationLists += `<h6>Encoded Categorical Columns (${data.cleaning_details.encoded_columns.length}):</h6>
                <ul class="small">
                    ${data.cleaning_details.encoded_columns.map(col => `<li>${col}</li>`).join('')}
                </ul>`;
            }
            
            if (data.cleaning_details.outliers_handled && Object.keys(data.cleaning_details.outliers_handled).length > 0) {
                operationLists += `<h6>Outliers Handled:</h6>
                <ul class="small">
                    ${Object.entries(data.cleaning_details.outliers_handled).map(([col, count]) => `
                        <li>${col}: ${count} outliers handled</li>
                    `).join('')}
                </ul>`;
            }
            
            // Main cleaning summary
            cleanResult.innerHTML = `
                <div class="alert alert-success">
                    <h5>Data Cleaning Results:</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>Rows:</strong> ${data.rows_before} → ${data.rows_after} (${data.rows_removed} removed)</p>
                            <p><strong>Columns:</strong> ${data.columns_before} → ${data.columns_after} (${data.columns_removed} removed)</p>
                            ${data.cleaning_details.duplicates_removed ? `<p><strong>Duplicates removed:</strong> ${data.cleaning_details.duplicates_removed}</p>` : ''}
                            ${data.cleaning_details.all_null_rows_dropped ? `<p><strong>All-null rows dropped:</strong> ${data.cleaning_details.all_null_rows_dropped}</p>` : ''}
                        </div>
                        <div class="col-md-6">
                            <div class="accordion" id="cleaningAccordion">
                                <div class="accordion-item">
                                    <h2 class="accordion-header" id="headingDetails">
                                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseDetails" aria-expanded="false" aria-controls="collapseDetails">
                                            Detailed Cleaning Operations
                                        </button>
                                    </h2>
                                    <div id="collapseDetails" class="accordion-collapse collapse" aria-labelledby="headingDetails" data-bs-parent="#cleaningAccordion">
                                        <div class="accordion-body">
                                            ${operationLists}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            showNotification('Success', 'Data cleaned successfully with advanced options', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to clean data', 'error');
        });
    }
    
    // Clean data button shows options modal
    cleanDataBtn.addEventListener('click', function() {
        showCleaningOptions();
    });
    
    // Train ML model
    mlForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const targetColumn = document.getElementById('target-column').value;
        let modelType = '';
        
        // Safely get the model type if the element exists
        const modelTypeElement = document.getElementById('model-type');
        if (modelTypeElement) {
            modelType = modelTypeElement.value;
        }
        
        if (!targetColumn) {
            showNotification('Error', 'Please select a target column', 'error');
            return;
        }
        
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            return;
        }
        
        showLoading();
        
        fetch('/ml', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: currentDatasetId,
                target_column: targetColumn,
                model_type: modelType // Include selected model type
            })
        })
        .then(response => {
            // Store response status for better error handling later
            const responseStatus = response.status;
            
            if (!response.ok) {
                // Try to parse error JSON for more details if possible
                return response.json().then(errorData => {
                    // Create an error object with status and details from response
                    const error = new Error(errorData.error || `Server returned ${responseStatus}: ${response.statusText}`);
                    error.status = responseStatus;
                    error.details = errorData;
                    throw error;
                }).catch(jsonError => {
                    // If JSON parsing fails, throw a simple error
                    const error = new Error(`Server returned ${responseStatus}: ${response.statusText}`);
                    error.status = responseStatus;
                    throw error;
                });
            }
            
            return response.json();
        })
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            // Show ML results
            const mlResult = document.getElementById('ml-result');
            const modelVisualizationsDiv = document.getElementById('model-visualizations');
            const modelComparisonDiv = document.getElementById('model-comparison');
            const modelInterpretationDiv = document.getElementById('model-interpretation');
            
            // Basic model results
            mlResult.innerHTML = `
                <div class="card mb-4">
                    <div class="card-header bg-primary bg-opacity-10">
                        <h5 class="mb-0 text-primary">
                            <i data-feather="award" class="me-2"></i>
                            Best Model Results
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Overview</h6>
                                <ul class="list-group list-group-flush">
                                    <li class="list-group-item bg-transparent">
                                        <strong>Problem Type:</strong> ${data.problem_type.charAt(0).toUpperCase() + data.problem_type.slice(1)}
                                    </li>
                                    <li class="list-group-item bg-transparent">
                                        <strong>Best Model:</strong> ${data.best_model.name}
                                    </li>
                                    <li class="list-group-item bg-transparent">
                                        <strong>${data.problem_type === 'classification' ? 'Accuracy' : 'R² Score'}:</strong> 
                                        ${data.problem_type === 'classification' 
                                          ? (data.best_model.performance.test_score * 100).toFixed(2) + '%' 
                                          : data.best_model.performance.test_r2.toFixed(4)}
                                    </li>
                                    <li class="list-group-item bg-transparent">
                                        <strong>Cross-Validation Score:</strong> 
                                        ${(data.best_model.performance.cv_mean * 100).toFixed(2)}% 
                                        (±${(data.best_model.performance.cv_std * 100).toFixed(2)}%)
                                    </li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6>Dataset Statistics</h6>
                                <ul class="list-group list-group-flush">
                                    <li class="list-group-item bg-transparent">
                                        <strong>Total Records:</strong> ${data.dataset_stats.total_records}
                                    </li>
                                    <li class="list-group-item bg-transparent">
                                        <strong>Features Used:</strong> ${data.dataset_stats.total_features}
                                    </li>
                                    <li class="list-group-item bg-transparent">
                                        <strong>Numeric Features:</strong> ${data.dataset_stats.numeric_features}
                                    </li>
                                    <li class="list-group-item bg-transparent">
                                        <strong>Categorical Features:</strong> ${data.dataset_stats.categorical_features}
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // Display Feature Importance
            if (data.model_explainability && data.model_explainability.feature_importance) {
                const featureImportanceHTML = `
                <div class="card mb-4">
                    <div class="card-header bg-primary bg-opacity-10">
                        <h5 class="mb-0 text-primary">
                            <i data-feather="bar-chart-2" class="me-2"></i>
                            Feature Importance
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="feature-importance-container">
                            ${Object.entries(data.model_explainability.feature_importance)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 15) // Show top 15 features
                                .map(([feature, importance]) => `
                                    <div class="feature-label">
                                        <span>${feature}</span>
                                        <span>${(importance * 100).toFixed(2)}%</span>
                                    </div>
                                    <div class="progress" style="height: 12px; margin-bottom: 8px;">
                                        <div class="progress-bar bg-primary" role="progressbar" 
                                            style="width: ${importance * 100}%" 
                                            aria-valuenow="${importance * 100}" 
                                            aria-valuemin="0" 
                                            aria-valuemax="100">
                                        </div>
                                    </div>
                                `).join('')}
                        </div>
                    </div>
                </div>`;
                
                modelVisualizationsDiv.innerHTML = featureImportanceHTML;
            }
            
            // Display Visualizations
            if (data.visualizations) {
                let visualizationsHTML = `
                <div class="card mb-4">
                    <div class="card-header bg-primary bg-opacity-10">
                        <h5 class="mb-0 text-primary">
                            <i data-feather="pie-chart" class="me-2"></i>
                            Model Visualizations
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="row">`;
                
                // Add each visualization
                if (data.visualizations.model_comparison) {
                    visualizationsHTML += `
                    <div class="col-md-12 mb-4">
                        <h6 class="mb-3">Model Comparison</h6>
                        <div class="text-center">
                            <img src="data:image/png;base64,${data.visualizations.model_comparison}" 
                                class="img-fluid border rounded" alt="Model Comparison" />
                        </div>
                    </div>`;
                }
                
                // Feature importance visualization
                if (data.visualizations.feature_importance) {
                    visualizationsHTML += `
                    <div class="col-md-12 mb-4">
                        <h6 class="mb-3">Top Feature Importance</h6>
                        <div class="text-center">
                            <img src="data:image/png;base64,${data.visualizations.feature_importance}" 
                                class="img-fluid border rounded" alt="Feature Importance" />
                        </div>
                    </div>`;
                }
                
                // Add problem-specific visualizations
                if (data.problem_type === 'classification') {
                    // Confusion Matrix for classification
                    if (data.visualizations.confusion_matrix) {
                        visualizationsHTML += `
                        <div class="col-md-6 mb-4">
                            <h6 class="mb-3">Confusion Matrix</h6>
                            <div class="text-center">
                                <img src="data:image/png;base64,${data.visualizations.confusion_matrix}" 
                                    class="img-fluid border rounded" alt="Confusion Matrix" />
                            </div>
                        </div>`;
                    }
                    
                    // ROC Curve for classification
                    if (data.visualizations.roc_curve) {
                        visualizationsHTML += `
                        <div class="col-md-6 mb-4">
                            <h6 class="mb-3">ROC Curve</h6>
                            <div class="text-center">
                                <img src="data:image/png;base64,${data.visualizations.roc_curve}" 
                                    class="img-fluid border rounded" alt="ROC Curve" />
                            </div>
                        </div>`;
                    }
                } else {
                    // Actual vs Predicted for regression
                    if (data.visualizations.actual_vs_predicted) {
                        visualizationsHTML += `
                        <div class="col-md-6 mb-4">
                            <h6 class="mb-3">Actual vs Predicted</h6>
                            <div class="text-center">
                                <img src="data:image/png;base64,${data.visualizations.actual_vs_predicted}" 
                                    class="img-fluid border rounded" alt="Actual vs Predicted" />
                            </div>
                        </div>`;
                    }
                    
                    // Residuals Plot for regression
                    if (data.visualizations.residuals_plot) {
                        visualizationsHTML += `
                        <div class="col-md-6 mb-4">
                            <h6 class="mb-3">Residuals Plot</h6>
                            <div class="text-center">
                                <img src="data:image/png;base64,${data.visualizations.residuals_plot}" 
                                    class="img-fluid border rounded" alt="Residuals Plot" />
                            </div>
                        </div>`;
                    }
                }
                
                // Decision Tree visualization if available
                if (data.visualizations.decision_tree) {
                    visualizationsHTML += `
                    <div class="col-md-12 mb-4">
                        <h6 class="mb-3">Decision Tree Visualization</h6>
                        <div class="text-center">
                            <img src="data:image/png;base64,${data.visualizations.decision_tree}" 
                                class="img-fluid border rounded" alt="Decision Tree" />
                        </div>
                    </div>`;
                }
                
                visualizationsHTML += `
                        </div>
                    </div>
                </div>`;
                
                // Add visualizations HTML
                modelVisualizationsDiv.innerHTML += visualizationsHTML;
            }
            
            // Display Model Comparison
            if (data.all_models) {
                let modelComparisonHTML = `
                <div class="card mb-4">
                    <div class="card-header bg-primary bg-opacity-10">
                        <h5 class="mb-0 text-primary">
                            <i data-feather="bar-chart" class="me-2"></i>
                            Model Comparison
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Model</th>
                                        <th>CV Score (Mean)</th>
                                        <th>CV Score (Std)</th>
                                        <th>${data.problem_type === 'classification' ? 'Test Accuracy' : 'Test R²'}</th>
                                        ${data.problem_type === 'classification' ? '<th>F1 Score</th>' : '<th>MSE</th>'}
                                    </tr>
                                </thead>
                                <tbody>`;
                
                // Add rows for each model
                Object.entries(data.all_models)
                    .filter(([name, model]) => !model.error) // Skip models with errors
                    .sort((a, b) => b[1].cv_mean - a[1].cv_mean) // Sort by CV score
                    .forEach(([name, model]) => {
                        const isSelected = name === data.best_model.name;
                        
                        modelComparisonHTML += `
                        <tr class="${isSelected ? 'table-primary' : ''}">
                            <td>${name} ${isSelected ? '<i data-feather="award" class="text-warning"></i>' : ''}</td>
                            <td>${(model.cv_mean * 100).toFixed(2)}%</td>
                            <td>±${(model.cv_std * 100).toFixed(2)}%</td>
                            <td>${data.problem_type === 'classification' 
                                ? (model.test_score * 100).toFixed(2) + '%' 
                                : model.test_r2.toFixed(4)}</td>
                            <td>${data.problem_type === 'classification' 
                                ? (model.f1_score).toFixed(4)
                                : model.test_mse.toFixed(4)}</td>
                        </tr>`;
                    });
                
                modelComparisonHTML += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>`;
                
                modelComparisonDiv.innerHTML = modelComparisonHTML;
            }
            
            // Display Model Interpretation
            if (data.model_explainability && data.model_explainability.interpretation) {
                modelInterpretationDiv.innerHTML = `
                <div class="card mb-4">
                    <div class="card-header bg-primary bg-opacity-10">
                        <h5 class="mb-0 text-primary">
                            <i data-feather="info" class="me-2"></i>
                            Model Interpretation
                        </h5>
                    </div>
                    <div class="card-body">
                        <p class="lead">${data.model_explainability.interpretation}</p>
                    </div>
                </div>`;
            }
            
            // Reinitialize Feather icons for new content
            feather.replace();
            
            showNotification('Success', 'Model trained successfully', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error during model training:', error);
            
            // Handle different error types with specific user-friendly messages
            let errorMessage = 'Failed to train model.';
            
            if (error.message && error.message.includes('timeout')) {
                errorMessage = 'Model training took too long and timed out. Try a simpler model or a smaller dataset.';
            } else if (error.status === 400) {
                errorMessage = 'Invalid request. Please check your model selection and target column.';
            } else if (error.message && error.message.includes('Server returned 400')) {
                errorMessage = 'Invalid request. Please check your model selection and target column.';
            } else if (error.status === 408 || (error.message && error.message.includes('Server returned 408'))) {
                errorMessage = 'Model training took too long and timed out. Try a simpler model or a smaller dataset.';
            }
            
            // Display error notification with more specific message
            showNotification('Error', errorMessage, 'error');
            
            // Also show error in results area for better visibility
            document.getElementById('ml-result').innerHTML = `
                <div class="alert alert-danger">
                    <h5><i data-feather="alert-circle"></i> Model Training Error</h5>
                    <p>${errorMessage}</p>
                    <hr>
                    <p class="mb-0">Try the following:</p>
                    <ul>
                        <li>Use a simpler model (Logistic Regression, Decision Tree)</li>
                        <li>Use a smaller dataset or fewer features</li>
                        <li>Make sure your target column is appropriate for the model type</li>
                    </ul>
                </div>
            `;
            
            // Reinitialize feather icons for the error message
            feather.replace();
        });
    });
    
    // Toggle mining method options
    document.querySelectorAll('input[name="mining-method"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const kmeans = this.value === 'kmeans';
            document.querySelector('.kmeans-only').style.display = kmeans ? 'block' : 'none';
        });
    });
    
    // Data mining
    mineForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const method = document.querySelector('input[name="mining-method"]:checked').value;
        const clusters = document.getElementById('clusters').value;
        
        // Get selected columns
        const selectedColumns = Array.from(document.querySelectorAll('input[name="mining-columns"]:checked'))
            .map(checkbox => checkbox.value);
        
        if (selectedColumns.length === 0) {
            showNotification('Error', 'Please select at least one column', 'error');
            return;
        }
        
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            return;
        }
        
        showLoading();
        
        fetch('/mine', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: currentDatasetId,
                method: method,
                columns: selectedColumns,
                n_clusters: parseInt(clusters)
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            // Show mining results
            const mineResult = document.getElementById('mine-result');
            
            if (method === 'apriori') {
                mineResult.innerHTML = `
                    <div class="result-section">
                        <h5>Association Rules:</h5>
                        <p>Found ${data.rules.length} rules</p>
                        <div class="table-responsive">
                            <table class="table table-sm rules-table">
                                <thead>
                                    <tr>
                                        <th>Antecedents</th>
                                        <th>Consequents</th>
                                        <th>Support</th>
                                        <th>Confidence</th>
                                        <th>Lift</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.rules.map(rule => `
                                        <tr>
                                            <td>${rule.antecedents.join(', ')}</td>
                                            <td>${rule.consequents.join(', ')}</td>
                                            <td>${rule.support.toFixed(3)}</td>
                                            <td>${rule.confidence.toFixed(3)}</td>
                                            <td>${rule.lift.toFixed(3)}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            } else {
                mineResult.innerHTML = `
                    <div class="result-section">
                        <h5>KMeans Clustering:</h5>
                        <p>Number of clusters: ${data.n_clusters}</p>
                        <p>Silhouette Score: ${data.silhouette_score.toFixed(3)}</p>
                        <h6 class="mt-3">Cluster Centers:</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Cluster</th>
                                        ${data.features.map(feature => `<th>${feature}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.cluster_centers.map((center, i) => `
                                        <tr>
                                            <td>Cluster ${i}</td>
                                            ${center.map(val => `<td>${val.toFixed(3)}</td>`).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                        <h6 class="mt-3">Cluster Sizes:</h6>
                        <ul>
                            ${Object.entries(data.cluster_sizes).map(([cluster, size]) => `
                                <li>Cluster ${cluster}: ${size} samples</li>
                            `).join('')}
                        </ul>
                    </div>
                `;
            }
            
            showNotification('Success', 'Data mining completed', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to perform data mining', 'error');
        });
    });
    
    // Toggle visualization options
    document.querySelectorAll('input[name="viz-type"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const isHeatmap = this.value === 'heatmap';
            
            document.querySelectorAll('.non-heatmap').forEach(el => {
                el.style.display = isHeatmap ? 'none' : 'block';
            });
            
            document.querySelector('.heatmap-only').style.display = isHeatmap ? 'block' : 'none';
        });
    });
    
    // Generate visualization
    visualizeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            return;
        }
        
        const vizType = document.querySelector('input[name="viz-type"]:checked').value;
        
        let requestData = {
            dataset_id: currentDatasetId,
            type: vizType
        };
        
        if (vizType === 'heatmap') {
            // Get selected columns
            const selectedColumns = Array.from(document.querySelectorAll('input[name="heatmap-columns"]:checked'))
                .map(checkbox => checkbox.value);
            
            if (selectedColumns.length < 2) {
                showNotification('Error', 'Please select at least two columns for heatmap', 'error');
                return;
            }
            
            requestData.columns = selectedColumns;
        } else {
            const xAxis = document.getElementById('x-axis').value;
            const yAxis = document.getElementById('y-axis').value;
            
            if (!xAxis || !yAxis) {
                showNotification('Error', 'Please select both X and Y axes', 'error');
                return;
            }
            
            requestData.x = xAxis;
            requestData.y = yAxis;
        }
        
        showLoading();
        
        fetch('/visualize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            // Show visualization
            const visualizationContainer = document.getElementById('visualization-container');
            visualizationContainer.innerHTML = data.html;
            
            // Make sure Plotly loads correctly by adding a small delay
            setTimeout(() => {
                // Check if visualization was properly rendered
                if (visualizationContainer.querySelector('.js-plotly-plot')) {
                    document.getElementById('visualization-export-options').style.display = 'block';
                    showNotification('Success', 'Visualization generated', 'success');
                } else {
                    console.log('Plotly visualization not found, rendering as iframe');
                    // If Plotly elements aren't found, it might be an iframe issue, force height
                    const iframes = visualizationContainer.querySelectorAll('iframe');
                    if (iframes.length > 0) {
                        iframes.forEach(iframe => {
                            iframe.style.width = '100%';
                            iframe.style.height = '600px';
                            iframe.style.border = 'none';
                        });
                        document.getElementById('visualization-export-options').style.display = 'block';
                        showNotification('Success', 'Visualization generated', 'success');
                    } else {
                        showNotification('Warning', 'Visualization may not have rendered properly', 'warning');
                    }
                }
            }, 500);
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to generate visualization', 'error');
        });
    });
    
    // Generate AI insights
    generateInsightsBtn.addEventListener('click', function() {
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            return;
        }
        
        showLoading();
        
        fetch('/ai-insight', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: currentDatasetId
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            // Show insights
            const insightsResult = document.getElementById('insights-result');
            
            // Format code blocks first
            let formattedInsights = data.insights.replace(/```python([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
            formattedInsights = formattedInsights.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
            
            // Replace **text** with highlighted spans
            formattedInsights = formattedInsights.replace(/\*\*(.*?)\*\*/g, '<span class="insights-highlight">$1</span>');
            
            // Format headings (lines that start with #)
            formattedInsights = formattedInsights.replace(/^# (.*?)$/gm, '<h3>$1</h3>');
            formattedInsights = formattedInsights.replace(/^## (.*?)$/gm, '<h4>$1</h4>');
            formattedInsights = formattedInsights.replace(/^### (.*?)$/gm, '<h5>$1</h5>');
            
            // Format numbered and bulleted lists
            formattedInsights = formattedInsights.replace(/^(\d+\.\s.*?)$/gm, '<li>$1</li>');
            formattedInsights = formattedInsights.replace(/^(\* .*?)$/gm, '<li>$1</li>');
            formattedInsights = formattedInsights.replace(/^(- .*?)$/gm, '<li>$1</li>');
            
            // Wrap consecutive list items with <ul> or <ol>
            formattedInsights = formattedInsights.replace(/<li>\d+\.\s([\s\S]*?)(?=<h|<div|$)/g, '<ol><li>$1</ol>');
            formattedInsights = formattedInsights.replace(/<li>[\*-]\s([\s\S]*?)(?=<h|<div|$)/g, '<ul><li>$1</ul>');
            
            // Split into sections (by headings) and wrap each in a styled div
            let sections = formattedInsights.split(/<h[3-5]/);
            let processedSections = '';
            
            if (sections.length > 1) {
                // First section might not have a heading
                if (sections[0].trim()) {
                    processedSections += `<div class="insights-section">${sections[0]}</div>`;
                }
                
                // Process the rest of the sections with their headings
                for (let i = 1; i < sections.length; i++) {
                    const headingLevel = formattedInsights.match(new RegExp(`<h([3-5]).*?${sections[i].substring(0, 20)}`))?.[1] || '3';
                    processedSections += `<div class="insights-section"><h${headingLevel}${sections[i]}</div>`;
                }
            } else {
                // If no clear sections, format paragraphs
                processedSections = `<div class="insights-section">${formattedInsights.replace(/\n\n/g, '</p><p>')}</div>`;
            }
            
            insightsResult.innerHTML = `
                <div class="result-section">
                    <h5>AI Insights:</h5>
                    <div class="insights-content">
                        ${processedSections}
                    </div>
                </div>
            `;
            
            showNotification('Success', 'AI insights generated', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to generate AI insights', 'error');
        });
    });
    
    // Anomaly Detection
    const anomalyDetectionForm = document.getElementById('anomaly-detection-form');
    const anomalyThreshold = document.getElementById('anomaly-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    
    // Update threshold value display when slider changes
    anomalyThreshold && anomalyThreshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
    });
    
    // Handle anomaly detection form submission
    anomalyDetectionForm && anomalyDetectionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            return;
        }
        
        const method = document.getElementById('anomaly-method').value;
        const threshold = parseFloat(document.getElementById('anomaly-threshold').value);
        
        showLoading();
        
        fetch('/detect-anomalies', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: currentDatasetId,
                method: method,
                threshold: threshold
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            const anomalyResults = document.getElementById('anomaly-results');
            
            // Create a results display
            let resultsHTML = `
                <div class="alert alert-info">
                    <strong>Detection Method:</strong> ${data.method}<br>
                    <strong>Total Anomalies:</strong> ${data.total_anomalies}<br>
                    <strong>Anomaly Percentage:</strong> ${data.anomaly_percentage.toFixed(2)}%
                </div>
            `;
            
            // If we have anomaly details, display them
            if (data.anomaly_details && data.anomaly_details.length > 0) {
                resultsHTML += `
                    <div class="card mt-3">
                        <div class="card-header">
                            <h6 class="mb-0">Anomaly Details</h6>
                        </div>
                        <div class="card-body p-0">
                            <div class="table-responsive">
                                <table class="table table-sm table-striped mb-0">
                                    <thead>
                                        <tr>
                                            <th>Row</th>
                                            <th>Details</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                `;
                
                data.anomaly_details.forEach(detail => {
                    let detailText = '';
                    
                    if (detail.columns) {
                        detailText = detail.columns.map(col => 
                            `${col.column}: ${col.value} ${col.z_score ? `(z-score: ${col.z_score.toFixed(2)})` : 
                             col.lower_bound ? `(outside range: ${col.lower_bound.toFixed(2)} - ${col.upper_bound.toFixed(2)})` : ''}`
                        ).join('<br>');
                    } else if (detail.anomaly_score) {
                        detailText = `Anomaly score: ${detail.anomaly_score.toFixed(4)}`;
                    } else if (detail.lof_score) {
                        detailText = `LOF score: ${detail.lof_score.toFixed(4)}`;
                    }
                    
                    resultsHTML += `
                        <tr>
                            <td>${detail.row_index}</td>
                            <td>${detailText}</td>
                        </tr>
                    `;
                });
                
                resultsHTML += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>`;
            }
            
            anomalyResults.innerHTML = resultsHTML;
            showNotification('Success', 'Anomaly detection completed', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to detect anomalies', 'error');
        });
    });
    
    // Data Quality Assessment
    const dataQualityBtn = document.getElementById('data-quality-btn');
    
    dataQualityBtn && dataQualityBtn.addEventListener('click', function() {
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            return;
        }
        
        showLoading();
        
        fetch('/assess-data-quality', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: currentDatasetId
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            const qualityResults = document.getElementById('quality-results');
            
            // Calculate overall quality color based on score
            let qualityColor = '#dc3545'; // red
            if (data.quality_score > 80) {
                qualityColor = '#28a745'; // green
            } else if (data.quality_score > 60) {
                qualityColor = '#ffc107'; // yellow
            } else if (data.quality_score > 40) {
                qualityColor = '#fd7e14'; // orange
            }
            
            // Create a results display
            let resultsHTML = `
                <div class="text-center mb-4">
                    <div class="progress-circle" style="--progress: ${data.quality_score}; --color: ${qualityColor};">
                        <div class="progress-circle-inner">
                            <div class="progress-value">${Math.round(data.quality_score)}</div>
                            <div class="progress-text">Quality Score</div>
                        </div>
                    </div>
                </div>
                
                <div class="card mb-3">
                    <div class="card-header">
                        <h6 class="mb-0">Dataset Overview</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4">
                                <strong>Rows:</strong> ${data.overall_statistics.row_count}
                            </div>
                            <div class="col-md-4">
                                <strong>Columns:</strong> ${data.overall_statistics.column_count}
                            </div>
                            <div class="col-md-4">
                                <strong>Missing Values:</strong> ${data.overall_statistics.missing_values.percentage.toFixed(2)}%
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // Add cleaning recommendations
            if (data.cleaning_recommendations && data.cleaning_recommendations.length > 0) {
                resultsHTML += `
                    <div class="card mb-3">
                        <div class="card-header">
                            <h6 class="mb-0">Cleaning Recommendations</h6>
                        </div>
                        <div class="card-body p-0">
                            <div class="list-group list-group-flush">
                `;
                
                data.cleaning_recommendations.forEach(rec => {
                    let icon = 'alert-triangle';
                    if (rec.issue.includes('high_')) {
                        icon = 'alert-octagon';
                    } else if (rec.issue.includes('duplicate')) {
                        icon = 'copy';
                    } else if (rec.issue.includes('inconsistent')) {
                        icon = 'edit';
                    }
                    
                    resultsHTML += `
                        <div class="list-group-item">
                            <div class="d-flex w-100 justify-content-between">
                                <h6 class="mb-1"><i data-feather="${icon}" class="me-2" style="width: 16px; height: 16px;"></i> ${rec.description}</h6>
                            </div>
                            <p class="mb-1">${rec.recommendation}</p>
                        </div>
                    `;
                });
                
                resultsHTML += `
                            </div>
                        </div>
                    </div>
                `;
            }
            
            qualityResults.innerHTML = resultsHTML;
            
            // Reinitialize feather icons in the new content
            feather.replace();
            
            showNotification('Success', 'Data quality assessment completed', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to assess data quality', 'error');
        });
    });
    
    // Analysis Recommendations
    const analysisRecommendationsBtn = document.getElementById('analysis-recommendations-btn');
    
    analysisRecommendationsBtn && analysisRecommendationsBtn.addEventListener('click', function() {
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            return;
        }
        
        showLoading();
        
        fetch('/analysis-recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: currentDatasetId
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            const recommendationsResults = document.getElementById('recommendations-results');
            
            // Create a results display
            let resultsHTML = '';
            
            // Recommended analyses
            if (data.recommended_analyses && data.recommended_analyses.length > 0) {
                resultsHTML += `
                    <div class="card mb-3">
                        <div class="card-header">
                            <h6 class="mb-0">Recommended Analyses</h6>
                        </div>
                        <div class="accordion" id="analysisAccordion">
                `;
                
                data.recommended_analyses.forEach((analysis, index) => {
                    resultsHTML += `
                        <div class="accordion-item">
                            <h2 class="accordion-header" id="heading${index}">
                                <button class="accordion-button ${index > 0 ? 'collapsed' : ''}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${index}" aria-expanded="${index === 0 ? 'true' : 'false'}" aria-controls="collapse${index}">
                                    ${analysis.analysis_type}
                                </button>
                            </h2>
                            <div id="collapse${index}" class="accordion-collapse collapse ${index === 0 ? 'show' : ''}" aria-labelledby="heading${index}" data-bs-parent="#analysisAccordion">
                                <div class="accordion-body">
                                    <p>${analysis.description}</p>
                                    <h6>Implementation Steps:</h6>
                                    <ol>
                                        ${analysis.implementation_steps.map(step => `<li>${step}</li>`).join('')}
                                    </ol>
                                    <div class="mt-3">
                                        <strong>Potential Insights:</strong> ${analysis.potential_insights}
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                resultsHTML += `
                        </div>
                    </div>
                `;
            }
            
            // Visualization recommendations
            if (data.recommended_visualizations && data.recommended_visualizations.length > 0) {
                resultsHTML += `
                    <div class="card mb-3">
                        <div class="card-header">
                            <h6 class="mb-0">Recommended Visualizations</h6>
                        </div>
                        <div class="card-body p-0">
                            <div class="list-group list-group-flush">
                `;
                
                data.recommended_visualizations.forEach(viz => {
                    let icon = 'bar-chart-2';
                    if (viz.visualization_type.toLowerCase().includes('scatter')) {
                        icon = 'activity';
                    } else if (viz.visualization_type.toLowerCase().includes('pie')) {
                        icon = 'pie-chart';
                    } else if (viz.visualization_type.toLowerCase().includes('line')) {
                        icon = 'trending-up';
                    } else if (viz.visualization_type.toLowerCase().includes('heat')) {
                        icon = 'grid';
                    }
                    
                    resultsHTML += `
                        <div class="list-group-item">
                            <div class="d-flex w-100 justify-content-between">
                                <h6 class="mb-1"><i data-feather="${icon}" class="me-2" style="width: 16px; height: 16px;"></i> ${viz.visualization_type}</h6>
                            </div>
                            <p class="mb-1">${viz.description}</p>
                            <small>Appropriate for: ${viz.appropriate_for.join(', ')}</small>
                        </div>
                    `;
                });
                
                resultsHTML += `
                            </div>
                        </div>
                    </div>
                `;
            }
            
            // Machine learning recommendations
            if (data.machine_learning_potential && data.machine_learning_potential.recommended_models) {
                resultsHTML += `
                    <div class="card mb-3">
                        <div class="card-header">
                            <h6 class="mb-0">Machine Learning Potential</h6>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <strong>Suitable for ML:</strong> ${data.machine_learning_potential.suitable_for_ml ? 'Yes' : 'No'}
                            </div>
                `;
                
                if (data.machine_learning_potential.recommended_models.length > 0) {
                    resultsHTML += `
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Model Type</th>
                                        <th>Appropriate For</th>
                                        <th>Expected Performance</th>
                                    </tr>
                                </thead>
                                <tbody>
                    `;
                    
                    data.machine_learning_potential.recommended_models.forEach(model => {
                        resultsHTML += `
                            <tr>
                                <td>${model.model_type}</td>
                                <td>${model.appropriate_for}</td>
                                <td>${model.expected_performance}</td>
                            </tr>
                        `;
                    });
                    
                    resultsHTML += `
                                </tbody>
                            </table>
                        </div>
                    `;
                }
                
                resultsHTML += `
                        </div>
                    </div>
                `;
            }
            
            recommendationsResults.innerHTML = resultsHTML;
            
            // Reinitialize feather icons in the new content
            feather.replace();
            
            showNotification('Success', 'Analysis recommendations generated', 'success');
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to generate analysis recommendations', 'error');
        });
    });
    
    // Chat with AI
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!currentDatasetId) {
            showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
            return;
        }
        
        const chatInput = document.getElementById('chat-input');
        const message = chatInput.value.trim();
        
        if (!message) {
            return;
        }
        
        // Add user message to chat
        const chatMessages = document.getElementById('chat-messages');
        const userMessage = document.createElement('div');
        userMessage.className = 'chat-message user';
        userMessage.textContent = message;
        chatMessages.appendChild(userMessage);
        
        // Clear input
        chatInput.value = '';
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        showLoading();
        
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: currentDatasetId,
                message: message
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showNotification('Error', data.error, 'error');
                return;
            }
            
            // Add AI response to chat
            const aiMessage = document.createElement('div');
            aiMessage.className = 'chat-message assistant';
            
            // Format the response
            let formattedResponse = data.response;
            
            // Format code blocks first
            formattedResponse = formattedResponse.replace(/```python([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
            formattedResponse = formattedResponse.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
            
            // Replace **text** with highlighted spans
            formattedResponse = formattedResponse.replace(/\*\*(.*?)\*\*/g, '<span class="insights-highlight">$1</span>');
            
            // Format headings (lines that start with #)
            formattedResponse = formattedResponse.replace(/^# (.*?)$/gm, '<h3>$1</h3>');
            formattedResponse = formattedResponse.replace(/^## (.*?)$/gm, '<h4>$1</h4>');
            formattedResponse = formattedResponse.replace(/^### (.*?)$/gm, '<h5>$1</h5>');
            
            // Format numbered and bulleted lists
            formattedResponse = formattedResponse.replace(/^(\d+\.\s.*?)$/gm, '<li>$1</li>');
            formattedResponse = formattedResponse.replace(/^(\* .*?)$/gm, '<li>$1</li>');
            formattedResponse = formattedResponse.replace(/^(- .*?)$/gm, '<li>$1</li>');
            
            // Wrap consecutive list items with <ul> or <ol>
            formattedResponse = formattedResponse.replace(/<li>\d+\.\s([\s\S]*?)(?=<h|<div|$)/g, '<ol><li>$1</ol>');
            formattedResponse = formattedResponse.replace(/<li>[\*-]\s([\s\S]*?)(?=<h|<div|$)/g, '<ul><li>$1</ul>');
            
            // Handle line breaks for the rest of the content
            formattedResponse = formattedResponse.replace(/\n\n/g, '<br><br>');
            formattedResponse = formattedResponse.replace(/\n/g, '<br>');
            
            aiMessage.innerHTML = formattedResponse;
            aiMessage.classList.add('formatted-response');
            chatMessages.appendChild(aiMessage);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            showNotification('Error', 'Failed to get AI response', 'error');
        });
    });
    
    // Natural Language Query form submission
    if (nlqueryForm) {
        nlqueryForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Check if we have a dataset
            if (!currentDatasetId) {
                showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
                return;
            }
            
            const queryInput = document.getElementById('nlquery-input');
            const query = queryInput.value.trim();
            
            if (!query) {
                showNotification('Error', 'Please enter a query', 'error');
                return;
            }
            
            // Get UI elements
            const resultsCard = document.getElementById('nlquery-results');
            const explanation = document.getElementById('nlquery-explanation');
            const codeBlock = document.getElementById('nlquery-code');
            const visualizationContainer = document.getElementById('nlquery-visualization-container');
            const visualization = document.getElementById('nlquery-visualization');
            const insights = document.getElementById('nlquery-insights');
            
            showLoading();
            
            // Send request to server
            fetch('/nl-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_id: currentDatasetId,
                    query: query
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.error) {
                    showNotification('Error', data.error, 'error');
                    return;
                }
                
                // Display the results
                resultsCard.classList.remove('d-none');
                
                // Show explanation
                explanation.innerHTML = data.explanation || 'No explanation provided.';
                
                // Show code
                codeBlock.textContent = data.code || 'No code generated.';
                
                // Show visualization if available
                if (data.visualization) {
                    visualizationContainer.classList.remove('d-none');
                    visualization.innerHTML = data.visualization;
                } else {
                    visualizationContainer.classList.add('d-none');
                    visualization.innerHTML = '';
                }
                
                // Show insights
                insights.innerHTML = data.insights || 'No insights provided.';
                
                // If there was an execution error, show it
                if (data.execution_error) {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'alert alert-danger mt-3';
                    errorDiv.innerHTML = `<strong>Execution Error:</strong> ${data.execution_error}`;
                    insights.appendChild(errorDiv);
                }
                
                // Initialize copy buttons
                document.querySelectorAll('.copy-code-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const targetId = this.getAttribute('data-target');
                        const textToCopy = document.getElementById(targetId).textContent;
                        
                        navigator.clipboard.writeText(textToCopy).then(() => {
                            const originalText = this.innerHTML;
                            this.innerHTML = '<i data-feather="check"></i> Copied!';
                            feather.replace();
                            
                            setTimeout(() => {
                                this.innerHTML = originalText;
                                feather.replace();
                            }, 2000);
                        }).catch(err => {
                            console.error('Failed to copy: ', err);
                        });
                    });
                });
                
                // Re-initialize Feather icons
                feather.replace();
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                showNotification('Error', 'Failed to process natural language query', 'error');
            });
        });
    }
    
    // Time Series Forecasting
    if (forecastForm) {
        forecastForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const dateColumn = document.getElementById('date-column').value;
            const targetColumn = document.getElementById('target-column').value;
            const forecastPeriods = document.getElementById('forecast-periods').value;
            const forecastMethod = document.getElementById('forecast-method').value;
            
            if (!dateColumn || !targetColumn) {
                showNotification('Error', 'Please select date and target columns', 'error');
                return;
            }
            
            if (!currentDatasetId) {
                showNotification('Error', 'No dataset found. Please upload a file first.', 'error');
                return;
            }
            
            showLoading();
            
            fetch('/timeseries-forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_id: currentDatasetId,
                    date_column: dateColumn,
                    target_column: targetColumn,
                    forecast_periods: parseInt(forecastPeriods),
                    method: forecastMethod
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.error) {
                    showNotification('Error', data.error, 'error');
                    return;
                }
                
                // Show forecast results
                const forecastResults = document.getElementById('forecast-results');
                forecastResults.classList.remove('d-none');
                
                // Display the forecast chart
                const forecastVisualization = document.getElementById('forecast-visualization');
                forecastVisualization.innerHTML = data.visualization;
                
                // Display model details
                const modelDetails = document.getElementById('forecast-model-details');
                modelDetails.innerHTML = '';
                
                // Add model info rows
                Object.entries(data.model_info).forEach(([key, value]) => {
                    const row = document.createElement('tr');
                    const keyCell = document.createElement('th');
                    const valueCell = document.createElement('td');
                    
                    keyCell.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    valueCell.textContent = value;
                    
                    row.appendChild(keyCell);
                    row.appendChild(valueCell);
                    modelDetails.appendChild(row);
                });
                
                // Display forecast values
                const forecastValues = document.getElementById('forecast-values');
                forecastValues.innerHTML = '';
                
                // Add forecast data rows
                data.forecast_values.forEach(value => {
                    const row = document.createElement('tr');
                    const dateCell = document.createElement('td');
                    const valueCell = document.createElement('td');
                    
                    dateCell.textContent = value.date;
                    valueCell.textContent = value.forecast.toFixed(2);
                    
                    row.appendChild(dateCell);
                    row.appendChild(valueCell);
                    forecastValues.appendChild(row);
                });
                
                showNotification('Success', 'Forecast generated successfully', 'success');
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                showNotification('Error', 'Failed to generate forecast', 'error');
            });
        });
    }
    
    // Dataset Comparison
    if (compareForm) {
        compareForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const dataset1 = document.getElementById('dataset1').value;
            const dataset2 = document.getElementById('dataset2').value;
            const dataset1Name = document.getElementById('dataset1-name').value || "Dataset 1";
            const dataset2Name = document.getElementById('dataset2-name').value || "Dataset 2";
            
            if (!dataset1 || !dataset2) {
                showNotification('Error', 'Please select two datasets to compare', 'error');
                return;
            }
            
            if (dataset1 === dataset2) {
                showNotification('Error', 'Please select two different datasets to compare', 'error');
                return;
            }
            
            showLoading();
            
            fetch('/compare-datasets', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_id1: dataset1,
                    dataset_id2: dataset2,
                    name1: dataset1Name,
                    name2: dataset2Name
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.error) {
                    showNotification('Error', data.error, 'error');
                    return;
                }
                
                // Show comparison results
                const comparisonResults = document.getElementById('comparison-results');
                comparisonResults.classList.remove('d-none');
                
                // Update dataset headings
                document.getElementById('dataset1-heading').textContent = dataset1Name;
                document.getElementById('dataset2-heading').textContent = dataset2Name;
                document.getElementById('dataset1-only-heading').textContent = `${dataset1Name} Only`;
                document.getElementById('dataset2-only-heading').textContent = `${dataset2Name} Only`;
                
                // Display key insights
                const comparisonInsights = document.getElementById('comparison-insights');
                comparisonInsights.innerHTML = data.insights.map(insight => `<p>• ${insight}</p>`).join('');
                
                // Display structure comparison
                const dataset1Structure = document.getElementById('dataset1-structure');
                const dataset2Structure = document.getElementById('dataset2-structure');
                
                dataset1Structure.innerHTML = `
                    <p><strong>Rows:</strong> ${data.structure.rows_1}</p>
                    <p><strong>Columns:</strong> ${data.structure.columns_1.length}</p>
                    <p><strong>Data Types:</strong> ${Object.entries(data.structure.dtypes_1).map(([col, type]) => `${col} (${type})`).join(', ')}</p>
                `;
                
                dataset2Structure.innerHTML = `
                    <p><strong>Rows:</strong> ${data.structure.rows_2}</p>
                    <p><strong>Columns:</strong> ${data.structure.columns_2.length}</p>
                    <p><strong>Data Types:</strong> ${Object.entries(data.structure.dtypes_2).map(([col, type]) => `${col} (${type})`).join(', ')}</p>
                `;
                
                // Display column comparison
                const commonColumns = document.getElementById('common-columns');
                const dataset1OnlyColumns = document.getElementById('dataset1-only-columns');
                const dataset2OnlyColumns = document.getElementById('dataset2-only-columns');
                
                commonColumns.innerHTML = '';
                dataset1OnlyColumns.innerHTML = '';
                dataset2OnlyColumns.innerHTML = '';
                
                data.columns.common_columns.forEach(column => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.textContent = column;
                    commonColumns.appendChild(li);
                });
                
                data.columns[`${dataset1Name}_only_columns`].forEach(column => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.textContent = column;
                    dataset1OnlyColumns.appendChild(li);
                });
                
                data.columns[`${dataset2Name}_only_columns`].forEach(column => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.textContent = column;
                    dataset2OnlyColumns.appendChild(li);
                });
                
                // Display visualizations
                const comparisonCharts = document.getElementById('comparison-charts');
                comparisonCharts.innerHTML = '';
                
                data.visualizations.forEach(viz => {
                    const vizContainer = document.createElement('div');
                    vizContainer.className = 'mb-4';
                    vizContainer.innerHTML = `
                        <h6 class="mb-2">${viz.title}</h6>
                        <div class="visualization-container">${viz.html}</div>
                    `;
                    comparisonCharts.appendChild(vizContainer);
                });
                
                showNotification('Success', 'Dataset comparison completed', 'success');
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                showNotification('Error', 'Failed to compare datasets', 'error');
            });
        });
    }
    
    // Initialize feather icons
    feather.replace();
    
    // ======= EXPORT FUNCTIONALITY =======
    
    // Dataset Export Functions
    function setupDatasetExportButtons() {
        // Basic export buttons
        const csvBtn = document.getElementById('export-csv');
        const excelBtn = document.getElementById('export-excel');
        const jsonBtn = document.getElementById('export-json');
        
        // Clean data export buttons
        const cleanCsvBtn = document.getElementById('export-clean-csv');
        const cleanExcelBtn = document.getElementById('export-clean-excel');
        
        // Original data export button
        const originalCsvBtn = document.getElementById('export-original-csv');
        
        // Function to handle export with type parameter
        function handleExport(format, type = 'current') {
            const datasetId = getCurrentDatasetId();
            if (!datasetId) {
                showNotification('Error', 'No dataset loaded to export', 'error');
                return;
            }
            window.location.href = `/export?format=${format}&dataset_id=${datasetId}&type=${type}`;
        }
        
        // Current data export
        if (csvBtn) {
            csvBtn.addEventListener('click', function(e) {
                e.preventDefault();
                handleExport('csv', 'current');
            });
        }
        
        if (excelBtn) {
            excelBtn.addEventListener('click', function(e) {
                e.preventDefault();
                handleExport('excel', 'current');
            });
        }
        
        if (jsonBtn) {
            jsonBtn.addEventListener('click', function(e) {
                e.preventDefault();
                handleExport('json', 'current');
            });
        }
        
        // Clean data export
        if (cleanCsvBtn) {
            cleanCsvBtn.addEventListener('click', function(e) {
                e.preventDefault();
                handleExport('csv', 'clean');
                showNotification('Processing', 'Cleaning and exporting data...', 'info');
            });
        }
        
        if (cleanExcelBtn) {
            cleanExcelBtn.addEventListener('click', function(e) {
                e.preventDefault();
                handleExport('excel', 'clean');
                showNotification('Processing', 'Cleaning and exporting data...', 'info');
            });
        }
        
        // Original data export
        if (originalCsvBtn) {
            originalCsvBtn.addEventListener('click', function(e) {
                e.preventDefault();
                handleExport('csv', 'original');
            });
        }
    }
    
    // AI Insights Export Functions
    function setupInsightsExportButtons() {
        const pdfBtn = document.getElementById('export-insights-pdf');
        const textBtn = document.getElementById('export-insights-text');
        const htmlBtn = document.getElementById('export-insights-html');
        
        if (pdfBtn) {
            pdfBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const insightsContent = document.getElementById('insights-result');
                if (!insightsContent || insightsContent.innerHTML.trim() === '') {
                    showNotification('Error', 'No insights available to export', 'error');
                    return;
                }
                
                // Export as PDF using jsPDF and html2canvas
                showLoading();
                html2canvas(insightsContent).then(canvas => {
                    const imgData = canvas.toDataURL('image/png');
                    const pdf = new jspdf.jsPDF({
                        orientation: 'portrait',
                        unit: 'mm',
                        format: 'a4'
                    });
                    
                    const imgWidth = 190;
                    const pageHeight = 297;
                    const imgHeight = (canvas.height * imgWidth) / canvas.width;
                    let heightLeft = imgHeight;
                    let position = 10;
                    
                    pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
                    heightLeft -= pageHeight;
                    
                    while (heightLeft > 0) {
                        position = heightLeft - imgHeight;
                        pdf.addPage();
                        pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
                        heightLeft -= pageHeight;
                    }
                    
                    pdf.save('data-insights.pdf');
                    hideLoading();
                    showNotification('Success', 'Insights exported to PDF', 'success');
                }).catch(err => {
                    hideLoading();
                    console.error('Error exporting to PDF:', err);
                    showNotification('Error', 'Failed to export insights', 'error');
                });
            });
        }
        
        if (textBtn) {
            textBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const insightsContent = document.getElementById('insights-result');
                if (!insightsContent || insightsContent.innerHTML.trim() === '') {
                    showNotification('Error', 'No insights available to export', 'error');
                    return;
                }
                
                // Export as plain text
                const text = insightsContent.innerText;
                const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
                saveAs(blob, 'data-insights.txt');
                showNotification('Success', 'Insights exported as text', 'success');
            });
        }
        
        if (htmlBtn) {
            htmlBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const insightsContent = document.getElementById('insights-result');
                if (!insightsContent || insightsContent.innerHTML.trim() === '') {
                    showNotification('Error', 'No insights available to export', 'error');
                    return;
                }
                
                // Export as HTML
                const html = `
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Data Analysis Insights</title>
                    <style>
                        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
                        h1, h2, h3 { color: #333; }
                        .container { max-width: 800px; margin: 0 auto; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Data Analysis Insights</h1>
                        <div class="insights-content">
                            ${insightsContent.innerHTML}
                        </div>
                        <p>Generated on ${new Date().toLocaleString()}</p>
                    </div>
                </body>
                </html>`;
                
                const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
                saveAs(blob, 'data-insights.html');
                showNotification('Success', 'Insights exported as HTML', 'success');
            });
        }
    }
    
    // Visualization Export Functions
    function setupVisualizationExportButtons() {
        const pngBtn = document.getElementById('export-visualization-png');
        const svgBtn = document.getElementById('export-visualization-svg');
        
        if (pngBtn) {
            pngBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const vizContainer = document.getElementById('visualization-container');
                if (!vizContainer || !vizContainer.querySelector('.js-plotly-plot')) {
                    showNotification('Error', 'No visualization available to export', 'error');
                    return;
                }
                
                // Get the plotly graph
                const plotDiv = vizContainer.querySelector('.js-plotly-plot');
                Plotly.downloadImage(plotDiv, {
                    format: 'png',
                    width: 1200,
                    height: 800,
                    filename: 'visualization'
                });
                
                showNotification('Success', 'Visualization exported as PNG', 'success');
            });
        }
        
        if (svgBtn) {
            svgBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const vizContainer = document.getElementById('visualization-container');
                if (!vizContainer || !vizContainer.querySelector('.js-plotly-plot')) {
                    showNotification('Error', 'No visualization available to export', 'error');
                    return;
                }
                
                // Get the plotly graph
                const plotDiv = vizContainer.querySelector('.js-plotly-plot');
                Plotly.downloadImage(plotDiv, {
                    format: 'svg',
                    width: 1200,
                    height: 800,
                    filename: 'visualization'
                });
                
                showNotification('Success', 'Visualization exported as SVG', 'success');
            });
        }
    }
    
    // Forecast Export Functions
    function setupForecastExportButtons() {
        const csvBtn = document.getElementById('export-forecast-csv');
        const chartBtn = document.getElementById('export-forecast-chart');
        const reportBtn = document.getElementById('export-forecast-report');
        
        if (csvBtn) {
            csvBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const forecastModelDetails = document.getElementById('forecast-model-details');
                const forecastTable = document.getElementById('forecast-data-table');
                
                if (!forecastTable || !forecastTable.querySelector('tbody')) {
                    showNotification('Error', 'No forecast data available to export', 'error');
                    return;
                }
                
                // Convert forecast table to CSV
                let csv = 'Date,Actual,Forecast,Lower Bound,Upper Bound\n';
                
                const rows = forecastTable.querySelectorAll('tbody tr');
                rows.forEach(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length >= 4) {
                        csv += `${cells[0].innerText},${cells[1].innerText},${cells[2].innerText},${cells[3].innerText},${cells[4].innerText}\n`;
                    }
                });
                
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
                saveAs(blob, 'forecast-data.csv');
                showNotification('Success', 'Forecast data exported as CSV', 'success');
            });
        }
        
        if (chartBtn) {
            chartBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const forecastViz = document.getElementById('forecast-visualization');
                if (!forecastViz || !forecastViz.querySelector('.js-plotly-plot')) {
                    showNotification('Error', 'No forecast chart available to export', 'error');
                    return;
                }
                
                // Get the plotly graph
                const plotDiv = forecastViz.querySelector('.js-plotly-plot');
                Plotly.downloadImage(plotDiv, {
                    format: 'png',
                    width: 1200,
                    height: 800,
                    filename: 'forecast-chart'
                });
                
                showNotification('Success', 'Forecast chart exported as PNG', 'success');
            });
        }
        
        if (reportBtn) {
            reportBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const forecastResults = document.getElementById('forecast-results');
                if (!forecastResults || forecastResults.innerHTML.trim() === '') {
                    showNotification('Error', 'No forecast report available to export', 'error');
                    return;
                }
                
                // Export as PDF using jsPDF and html2canvas
                showLoading();
                html2canvas(forecastResults).then(canvas => {
                    const imgData = canvas.toDataURL('image/png');
                    const pdf = new jspdf.jsPDF({
                        orientation: 'portrait',
                        unit: 'mm',
                        format: 'a4'
                    });
                    
                    const imgWidth = 190;
                    const pageHeight = 297;
                    const imgHeight = (canvas.height * imgWidth) / canvas.width;
                    let heightLeft = imgHeight;
                    let position = 10;
                    
                    pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
                    heightLeft -= pageHeight;
                    
                    while (heightLeft > 0) {
                        position = heightLeft - imgHeight;
                        pdf.addPage();
                        pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
                        heightLeft -= pageHeight;
                    }
                    
                    pdf.save('forecast-report.pdf');
                    hideLoading();
                    showNotification('Success', 'Forecast report exported to PDF', 'success');
                }).catch(err => {
                    hideLoading();
                    console.error('Error exporting to PDF:', err);
                    showNotification('Error', 'Failed to export forecast report', 'error');
                });
            });
        }
    }
    
    // Helper function to get current dataset ID
    function getCurrentDatasetId() {
        // First try from global variable (set during file upload)
        if (currentDatasetId) {
            return currentDatasetId;
        }
        
        // Try to get from session storage as fallback
        let datasetId = sessionStorage.getItem('current_dataset_id');
        
        // If not found in session storage, try to get from active tab
        if (!datasetId) {
            // Try to extract from the URL query params or other UI elements
            const urlParams = new URLSearchParams(window.location.search);
            datasetId = urlParams.get('dataset_id');
        }
        
        return datasetId;
    }
    
    // Setup export functionality when content is loaded
    function setupExportFunctionality() {
        // Setup Dataset Export
        setupDatasetExportButtons();
        
        // Setup AI Insights Export
        setupInsightsExportButtons();
        
        // Setup Visualization Export
        setupVisualizationExportButtons();
        
        // Setup Forecast Export
        setupForecastExportButtons();
        
        // Show visualization export options when a visualization is created
        const vizContainer = document.getElementById('visualization-container');
        if (vizContainer) {
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        // Check if a plotly plot was added
                        if (vizContainer.querySelector('.js-plotly-plot')) {
                            document.getElementById('visualization-export-options').style.display = 'block';
                        }
                    }
                });
            });
            
            observer.observe(vizContainer, { childList: true, subtree: true });
        }
        
        // Show insights export options when insights are generated
        const insightsResult = document.getElementById('insights-result');
        if (insightsResult) {
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        // Check if content was added
                        if (insightsResult.innerHTML.trim() !== '') {
                            document.getElementById('insights-export-dropdown').style.display = 'block';
                        }
                    }
                });
            });
            
            observer.observe(insightsResult, { childList: true });
        }
    }
    
    // Call setup function
    setupExportFunctionality();
    
    // Show export dropdown in data tab when data is loaded
    function showDataExportOptions() {
        const exportDropdown = document.getElementById('exportDropdown');
        if (exportDropdown) {
            exportDropdown.style.display = 'inline-block';
        }
    }
    
    // Add dataset load event listener to show export buttons
    document.addEventListener('datasetLoaded', function(e) {
        showDataExportOptions();
    });
});
