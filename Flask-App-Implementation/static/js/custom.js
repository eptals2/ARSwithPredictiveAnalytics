document.addEventListener("DOMContentLoaded", function () {
    const resumeForm = document.getElementById("resumeForm");
    const extractBtn = document.getElementById("extractBtn");
    const compareBtn = document.getElementById("compareBtn");
    const loadingSpinner = document.getElementById("loadingSpinner");
    const resultsContainer = document.getElementById("resultsContainer");
    const candidateList = document.getElementById("candidateList");
    const comparisonCard = document.getElementById("comparisonCard");
    const comparisonContainer = document.getElementById("comparisonContainer");
    const assessmentCard = document.getElementById("assessmentCard");
    const assessmentContainer = document.getElementById("assessmentContainer");

    let analysisResults = [];

    // Hide loading spinner initially
    loadingSpinner.style.display = "none";

    // Handle form submission for resume upload
    resumeForm.addEventListener("submit", function (e) {
        e.preventDefault();

        const fileInput = document.getElementById("resumeFiles");
        if (!fileInput.files.length) {
            alert("Please select at least one resume file.");
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < fileInput.files.length; i++) {
            formData.append("resumes[]", fileInput.files[i]);
        }

        // Show loading spinner
        loadingSpinner.style.display = "block";
        resultsContainer.classList.add("hidden");
        comparisonCard.classList.add("hidden");
        assessmentCard.classList.add("hidden");

        // Send request to extract information
        fetch("/extract", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.status === "success") {
                    analysisResults = data.results;
                    displayResults(analysisResults);
                    resultsContainer.classList.remove("hidden");

                    // Show compare button only if there are multiple results
                    if (analysisResults.length > 1) {
                        compareBtn.classList.remove("hidden");
                    } else {
                        compareBtn.classList.add("hidden");
                    }
                } else {
                    alert("Error: " + data.error);
                }
            })
            .catch((error) => {
                console.error("Error:", error);
                alert("An error occurred while processing the resumes.");
            })
            .finally(() => {
                loadingSpinner.style.display = "none";
            });
    });

    // Handle compare button click
    compareBtn.addEventListener("click", function () {
        if (analysisResults.length <= 1) {
            alert("You need at least two candidates to compare.");
            return;
        }

        // Show loading spinner
        loadingSpinner.style.display = "block";
        comparisonCard.classList.add("hidden");

        // Send request to compare candidates
        fetch("/compare", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ results: analysisResults }),
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.status === "success") {
                    displayComparison(data.comparison);
                    comparisonCard.classList.remove("hidden");
                    assessmentCard.classList.add("hidden");
                } else {
                    alert("Error: " + data.error);
                }
            })
            .catch((error) => {
                console.error("Error:", error);
                alert("An error occurred while comparing candidates.");
            })
            .finally(() => {
                loadingSpinner.style.display = "none";
            });
    });

    // Function to display results for all candidates
    function displayResults(results) {
        candidateList.innerHTML = "";

        // Sort results by overall score in descending order
        results.sort((a, b) => b.assessment.overall_score - a.assessment.overall_score);

        results.forEach((result, index) => {
            const candidateCard = document.createElement("div");
            candidateCard.className = "candidate-card";

            const cardHeader = document.createElement("div");
            cardHeader.className = "candidate-card-header";
            cardHeader.textContent = result.filename;

            const cardBody = document.createElement("div");
            cardBody.className = "candidate-card-body";

            // Display score, suitability and position
            const scoreDiv = document.createElement("div");
            scoreDiv.className = "mb-3";
            
            const positions = result.entities.job_position || [];
            const positionText = positions.length > 0 ? `<strong>Suitable Position:</strong> ${positions.join(", ")}<br>` : "";
            
            scoreDiv.innerHTML = `
                ${positionText}
                <strong>Score:</strong> ${result.assessment.overall_score}% - 
                <span class="${result.assessment.suitability.toLowerCase().replace(" ", "-")}">${result.assessment.suitability}</span>
            `;

            // Create view details button
            const viewDetailsBtn = document.createElement("button");
            viewDetailsBtn.className = "btn btn-outline-primary btn-sm";
            viewDetailsBtn.textContent = "View Details";
            viewDetailsBtn.addEventListener("click", function () {
                displayAssessment(result.assessment, result.entities);
                assessmentCard.classList.remove("hidden");
                comparisonCard.classList.add("hidden");
            });

            cardBody.appendChild(scoreDiv);
            cardBody.appendChild(viewDetailsBtn);

            candidateCard.appendChild(cardHeader);
            candidateCard.appendChild(cardBody);
            candidateList.appendChild(candidateCard);
        });
    }

    // Function to display assessment results for a single candidate
    function displayAssessment(assessment, entities) {
        assessmentContainer.innerHTML = "";

        // Create score section
        const scoreSection = document.createElement("div");
        scoreSection.className = "mb-4";
        
        const positions = entities.job_position || [];
        const positionText = positions.length > 0 ? `<p><strong>Predicted Suitable Job:</strong> ${positions.join(", ")}</p>` : "";

        scoreSection.innerHTML = `
            <div class="mb-3 p-2 border-bottom">
            ${positionText}
            <p><strong>Suitability Percentage:</strong> ${assessment.overall_score}%</p>
            <p><strong>Suitability Status:</strong> <span class="${assessment.suitability.toLowerCase().replace(" ", "-")}">${assessment.suitability}</span></p>
            <!--p><strong>Recommendation:</strong> $//{assessment.recommendation}</!--p>
            </div>
        `;
        assessmentContainer.appendChild(scoreSection);

        // Create entities section
        const entitiesSection = document.createElement("div");
        entitiesSection.className = "mb-4";
        entitiesSection.innerHTML = "<h4>Extracted Details</h4>";

        // Display entities
        const entityGroups = {
            "Education": ["education_level", "course"],
            "Skills": ["soft_skills", "hard_skills"],
            "Experience": ["experience"],
            "Certifications": ["certification"]
        };

        for (const [groupName, entityKeys] of Object.entries(entityGroups)) {
            const groupDiv = document.createElement("div");
            groupDiv.className = "mb-3";
            groupDiv.innerHTML = `<h5>${groupName}</h5>`;

            entityKeys.forEach(key => {
                if (entities[key] && entities[key].length > 0) {
                    const entityDiv = document.createElement("div");
                    entityDiv.innerHTML = `<p><strong>${key.replace("_", " ").replace(/\b\w/g, l => l.toUpperCase())}:</strong> ${entities[key].join(", ")}</p>`;
                    groupDiv.appendChild(entityDiv);
                }
            });

            entitiesSection.appendChild(groupDiv);
        }

        assessmentContainer.appendChild(entitiesSection);
    }

    // Function to display comparison results
    function displayComparison(comparison) {
        comparisonContainer.innerHTML = "";

        // Create best candidate alert
        const bestCandidateAlert = document.createElement("div");
        bestCandidateAlert.className = "alert alert-success";
        bestCandidateAlert.innerHTML = `
            <strong>Best Candidate:</strong> ${comparison.best_candidate.name} with a score of ${comparison.best_candidate.score}%
        `;
        comparisonContainer.appendChild(bestCandidateAlert);

        // Create comparison table
        const table = document.createElement("table");
        table.className = "comparison-table";

        // Create table header
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");

        const categoryHeader = document.createElement("th");
        categoryHeader.textContent = "Candidate";
        headerRow.appendChild(categoryHeader);

        const scoreHeader = document.createElement("th");
        scoreHeader.textContent = "Score";
        headerRow.appendChild(scoreHeader);

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create table body
        const tbody = document.createElement("tbody");

        comparison.candidates.forEach((candidate, index) => {
            const row = document.createElement("tr");
            
            const nameCell = document.createElement("td");
            nameCell.textContent = candidate;
            
            const scoreCell = document.createElement("td");
            scoreCell.textContent = comparison.overall_scores[index] + "%";

            if (index === comparison.best_candidate.index) {
                row.className = "best-candidate";
            }

            row.appendChild(nameCell);
            row.appendChild(scoreCell);
            tbody.appendChild(row);
        });

        table.appendChild(tbody);
        comparisonContainer.appendChild(table);
    }
});
