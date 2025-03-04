document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    const processTextBtn = document.getElementById('processTextBtn');
    const clearBtn = document.getElementById('clearBtn');
    const textContainer = document.getElementById('textContainer');
    const exportBtn = document.getElementById('exportBtn');
    const exportOutput = document.getElementById('exportOutput');
    const exportFormat = document.getElementById('exportFormat');
    
    let currentEntityType = null;
    let isLabeling = false;
    let labeledTokens = [];
    let allTokens = [];
    let entityCounts = {
        AGE: 0,
        GENDER: 0,
        ADDRESS: 0,
        SOFT_SKILL: 0,
        HARD_SKILL: 0,
        EDUCATION_LEVEL: 0,
        COURSE: 0,
        EXPERIENCE: 0,
        CERTIFICATION: 0
    };
    
    // Process text into tokens
    processTextBtn.addEventListener('click', function() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to process.');
            return;
        }
        
        // Reset previous data
        resetLabeling();
        
        // Simple tokenization (split by whitespace and punctuation)
        const tokens = text.split(/(\s+|[,.!?;:()[\]{}'""-])/g)
            .filter(token => token.trim() !== '');
        
        allTokens = tokens.map((token, index) => ({
            id: index,
            text: token,
            entity: null,
            isWhitespace: /^\s+$/.test(token)
        }));
        
        renderTokens();
    });
    
    // Clear all data
    clearBtn.addEventListener('click', function() {
        resetLabeling();
        textInput.value = '';
        textContainer.innerHTML = '';
        exportOutput.value = '';
    });
    
    // Handle keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Number keys 1-9 for entity selection
        if (e.key >= '1' && e.key <= '9') {
            const entityIndex = parseInt(e.key) - 1;
            const entityRadios = document.querySelectorAll('input[name="entityType"]');
            if (entityIndex < entityRadios.length) {
                entityRadios[entityIndex].checked = true;
                currentEntityType = entityRadios[entityIndex].value;
                updateSelectedEntityType();
            }
        }
        // Enter key to finish labeling
        else if (e.key === 'Enter' && isLabeling) {
            finishLabeling();
        }
        // Escape key to cancel current selection
        else if (e.key === 'Escape' && isLabeling) {
            cancelLabeling();
        }
    });
    
    // Handle entity type selection via radio buttons
    document.querySelectorAll('input[name="entityType"]').forEach(radio => {
        radio.addEventListener('change', function() {
            currentEntityType = this.value;
            updateSelectedEntityType();
        });
    });
    
    // Export labeled data
    exportBtn.addEventListener('click', function() {
        if (labeledTokens.length === 0) {
            alert('No labeled data to export.');
            return;
        }
        
        const format = exportFormat.value;
        if (format === 'conll') {
            exportOutput.value = exportToCoNLL();
        } else if (format === 'json') {
            exportOutput.value = exportToJSON();
        }
    });
    
    // Render tokens in the text container
    function renderTokens() {
        textContainer.innerHTML = '';
        
        allTokens.forEach(token => {
            if (token.isWhitespace) {
                // For whitespace tokens, just add the whitespace
                const spaceNode = document.createTextNode(token.text);
                textContainer.appendChild(spaceNode);
            } else {
                // For non-whitespace tokens, create clickable spans
                const tokenSpan = document.createElement('span');
                tokenSpan.className = 'token';
                tokenSpan.textContent = token.text;
                tokenSpan.dataset.id = token.id;
                
                // If token has an entity, add styling
                if (token.entity) {
                    tokenSpan.classList.add('labeled-token');
                    tokenSpan.classList.add(token.entity);
                    tokenSpan.title = token.entity;
                }
                
                // Add click event for labeling
                tokenSpan.addEventListener('click', function() {
                    if (!currentEntityType) {
                        alert('Please select an entity type first.');
                        return;
                    }
                    
                    // If token already has this entity, remove it
                    if (token.entity === currentEntityType) {
                        token.entity = null;
                        updateEntityCount(currentEntityType, -1);
                        renderTokens();
                        updateLabeledTokens();
                    } 
                    // If token has a different entity, change it
                    else if (token.entity) {
                        updateEntityCount(token.entity, -1);
                        token.entity = currentEntityType;
                        updateEntityCount(currentEntityType, 1);
                        renderTokens();
                        updateLabeledTokens();
                    } 
                    // If token has no entity, add the current one
                    else {
                        token.entity = currentEntityType;
                        updateEntityCount(currentEntityType, 1);
                        renderTokens();
                        updateLabeledTokens();
                    }
                });
                
                textContainer.appendChild(tokenSpan);
            }
        });
    }
    
    // Update the selected entity type visual indicator
    function updateSelectedEntityType() {
        document.querySelectorAll('.selected-entity').forEach(el => {
            el.classList.remove('selected-entity');
        });
        
        if (currentEntityType) {
            const selectedRadio = document.querySelector(`input[value="${currentEntityType}"]`);
            if (selectedRadio) {
                const label = selectedRadio.closest('.form-check');
                if (label) {
                    label.classList.add('selected-entity');
                }
            }
        }
    }
    
    // Update entity count display
    function updateEntityCount(entityType, change) {
        entityCounts[entityType] += change;
        const countElement = document.getElementById(`${entityType.toLowerCase()}Count`);
        if (countElement) {
            countElement.textContent = `(${entityCounts[entityType]})`;
        }
    }
    
    // Update the labeled tokens array
    function updateLabeledTokens() {
        labeledTokens = allTokens.filter(token => token.entity !== null && !token.isWhitespace);
    }
    
    // Finish the current labeling session
    function finishLabeling() {
        isLabeling = false;
        updateSelectedEntityType();
    }
    
    // Cancel the current labeling session
    function cancelLabeling() {
        isLabeling = false;
        currentEntityType = null;
        document.querySelectorAll('input[name="entityType"]').forEach(radio => {
            radio.checked = false;
        });
        updateSelectedEntityType();
    }
    
    // Reset all labeling data
    function resetLabeling() {
        isLabeling = false;
        currentEntityType = null;
        labeledTokens = [];
        allTokens = [];
        
        // Reset entity counts
        Object.keys(entityCounts).forEach(key => {
            entityCounts[key] = 0;
            const countElement = document.getElementById(`${key.toLowerCase()}Count`);
            if (countElement) {
                countElement.textContent = '(0)';
            }
        });
        
        document.querySelectorAll('input[name="entityType"]').forEach(radio => {
            radio.checked = false;
        });
        
        updateSelectedEntityType();
    }
    
    // Export data in CoNLL format
    function exportToCoNLL() {
        let output = '';
        let sentenceId = 1;
        let tokenId = 1;
        
        allTokens.forEach(token => {
            if (token.isWhitespace) {
                // Skip whitespace tokens in CoNLL format
                return;
            }
            
            // If token has a newline, start a new sentence
            if (token.text.includes('\n')) {
                output += '\n';
                sentenceId++;
                tokenId = 1;
                return;
            }
            
            const entityTag = token.entity ? token.entity : 'O';
            // Format: token_id token_text entity_tag
            output += `${tokenId}\t${token.text}\t${entityTag}\n`;
            tokenId++;
        });
        
        return output;
    }
    
    // Export data in JSON format
    function exportToJSON() {
        const sentences = [];
        let currentSentence = {
            tokens: [],
            entities: []
        };
        
        let entityStartIndex = -1;
        let entityType = null;
        let tokenIndex = 0;
        
        allTokens.forEach(token => {
            if (token.isWhitespace) {
                // Skip whitespace tokens for entity calculation
                return;
            }
            
            // Add token to current sentence
            currentSentence.tokens.push(token.text);
            
            // If token has an entity, track it
            if (token.entity) {
                if (entityType !== token.entity) {
                    // If we were tracking an entity, add it to the entities list
                    if (entityType !== null && entityStartIndex !== -1) {
                        currentSentence.entities.push({
                            start: entityStartIndex,
                            end: tokenIndex - 1,
                            type: entityType
                        });
                    }
                    
                    // Start tracking a new entity
                    entityType = token.entity;
                    entityStartIndex = tokenIndex;
                }
            } else {
                // If token has no entity but we were tracking one, add it to the entities list
                if (entityType !== null && entityStartIndex !== -1) {
                    currentSentence.entities.push({
                        start: entityStartIndex,
                        end: tokenIndex - 1,
                        type: entityType
                    });
                    
                    // Reset entity tracking
                    entityType = null;
                    entityStartIndex = -1;
                }
            }
            
            tokenIndex++;
            
            // If token ends with a period, question mark, or exclamation point, start a new sentence
            if (/[.!?]$/.test(token.text)) {
                // Add any remaining entity
                if (entityType !== null && entityStartIndex !== -1) {
                    currentSentence.entities.push({
                        start: entityStartIndex,
                        end: tokenIndex - 1,
                        type: entityType
                    });
                    
                    // Reset entity tracking
                    entityType = null;
                    entityStartIndex = -1;
                }
                
                sentences.push(currentSentence);
                currentSentence = {
                    tokens: [],
                    entities: []
                };
                tokenIndex = 0;
            }
        });
        
        // Add any remaining sentence
        if (currentSentence.tokens.length > 0) {
            // Add any remaining entity
            if (entityType !== null && entityStartIndex !== -1) {
                currentSentence.entities.push({
                    start: entityStartIndex,
                    end: tokenIndex - 1,
                    type: entityType
                });
            }
            
            sentences.push(currentSentence);
        }
        return JSON.stringify({
            data: sentences
        }, null, 2);
    }
});