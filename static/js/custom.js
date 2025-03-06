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

        results.forEach((result, index) => {
            const candidateCard = document.createElement("div");
            candidateCard.className = "candidate-card";

            const cardHeader = document.createElement("div");
            cardHeader.className = "candidate-card-header";
            cardHeader.textContent = result.filename;

            const cardBody = document.createElement("div");
            cardBody.className = "candidate-card-body";

            // Display score and suitability
            const scoreDiv = document.createElement("div");
            scoreDiv.className = "mb-3";
            scoreDiv.innerHTML = `
            <strong>Score:</strong> ${result.assessment.overall_score}% - 
            <span class="${result.assessment.suitability
                .toLowerCase()
                .replace(" ", "-")}">${result.assessment.suitability}</span>
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
        categoryHeader.textContent = "Category";
        headerRow.appendChild(categoryHeader);

        comparison.candidates.forEach((candidate) => {
            const candidateHeader = document.createElement("th");
            candidateHeader.textContent = candidate;
            headerRow.appendChild(candidateHeader);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create table body
        const tbody = document.createElement("tbody");

        // Overall score row
        const overallRow = document.createElement("tr");
        const overallLabel = document.createElement("td");
        overallLabel.textContent = "Overall Score";
        overallRow.appendChild(overallLabel);

        comparison.overall_scores.forEach((score, index) => {
            const scoreCell = document.createElement("td");
            scoreCell.textContent = score + "%";

            if (index === comparison.best_candidate.index) {
                scoreCell.className = "best-candidate";
            }

            overallRow.appendChild(scoreCell);
        });

        tbody.appendChild(overallRow);

        // Category scores rows
        const categories = [
            { key: "soft_skills", title: "Soft Skills" },
            { key: "hard_skills", title: "Hard Skills" },
            { key: "education", title: "Education" },
            { key: "experience", title: "Experience" },
            { key: "certification", title: "Certification" },
        ];

        categories.forEach((category) => {
            const categoryRow = document.createElement("tr");
            const categoryLabel = document.createElement("td");
            categoryLabel.textContent = category.title;
            categoryRow.appendChild(categoryLabel);

            comparison.category_scores[category.key].forEach((score) => {
                const scoreCell = document.createElement("td");
                scoreCell.textContent = score + "%";
                categoryRow.appendChild(scoreCell);
            });

            tbody.appendChild(categoryRow);
        });

        // Suitability row
        const suitabilityRow = document.createElement("tr");
        const suitabilityLabel = document.createElement("td");
        suitabilityLabel.textContent = "Suitability";
        suitabilityRow.appendChild(suitabilityLabel);

        comparison.suitability.forEach((suitability) => {
            const suitabilityCell = document.createElement("td");
            suitabilityCell.textContent = suitability;
            suitabilityRow.appendChild(suitabilityCell);
        });

        tbody.appendChild(suitabilityRow);

        table.appendChild(tbody);
        comparisonContainer.appendChild(table);

        // Create recommendations section
        const recommendationsDiv = document.createElement("div");
        recommendationsDiv.className = "mt-4";
        recommendationsDiv.innerHTML = "<h5>Recommendations:</h5>";

        const recommendationsList = document.createElement("ul");
        comparison.candidates.forEach((candidate, index) => {
            const recommendationItem = document.createElement("li");
            recommendationItem.innerHTML = `<strong>${candidate}:</strong> ${comparison.recommendations[index]}`;
            recommendationsList.appendChild(recommendationItem);
        });

        recommendationsDiv.appendChild(recommendationsList);
        comparisonContainer.appendChild(recommendationsDiv);
    }

    // Function to display assessment results for a single candidate
    function displayAssessment(assessment, entities) {
        assessmentContainer.innerHTML = "";

        // Overall score
        const scoreDiv = document.createElement("div");
        scoreDiv.className = "assessment-score";
        scoreDiv.textContent = assessment.overall_score + "%";

        // Suitability level
        const suitabilityDiv = document.createElement("div");
        suitabilityDiv.className = "text-center mb-4";
        suitabilityDiv.innerHTML = `<h4>Suitability: <span class="${assessment.suitability
            .toLowerCase()
            .replace(" ", "-")}">${assessment.suitability}</span></h4>`;

        assessmentContainer.appendChild(scoreDiv);
        assessmentContainer.appendChild(suitabilityDiv);

        
        // Category scores
        const categoryScoresDiv = document.createElement("div");
        categoryScoresDiv.className = "mb-4";
        categoryScoresDiv.innerHTML = "<h5>Category Scores:</h5>";

        const categories = [
            { key: "soft_skills", title: "Soft Skills" },
            { key: "hard_skills", title: "Hard Skills" },
            { key: "education", title: "Education" },
            { key: "experience", title: "Experience" },
            { key: "certification", title: "Certification" },
        ];

        categories.forEach((category) => {
            const score = assessment.category_scores[category.key];

            const categoryDiv = document.createElement("div");
            categoryDiv.className = "assessment-category";

            const categoryTitle = document.createElement("div");
            categoryTitle.className = "assessment-category-title";
            categoryTitle.textContent = `${category.title}: ${score}%`;

            const progressDiv = document.createElement("div");
            progressDiv.className = "progress";

            let progressClass = "bg-danger";
            if (score >= 70) {
                progressClass = "bg-success";
            } else if (score >= 50) {
                progressClass = "bg-warning";
            }

            progressDiv.innerHTML = `<div class="progress-bar ${progressClass}" role="progressbar" style="width: ${score}%" aria-valuenow="${score}" aria-valuemin="0" aria-valuemax="100"></div>`;

            categoryDiv.appendChild(categoryTitle);
            categoryDiv.appendChild(progressDiv);
            categoryScoresDiv.appendChild(categoryDiv);
        });

        assessmentContainer.appendChild(categoryScoresDiv);

        // Strengths
        const strengthsDiv = document.createElement("div");
        strengthsDiv.className = "mb-3";
        strengthsDiv.innerHTML = `<h5>Strengths:</h5><ul>${assessment.strengths
            .map((s) => `<li>${s}</li>`)
            .join("")}</ul>`;
        assessmentContainer.appendChild(strengthsDiv);

        // Areas for improvement
        const areasDiv = document.createElement("div");
        areasDiv.className = "mb-3";
        areasDiv.innerHTML = `<h5>Areas for Improvement:</h5><ul>${assessment.areas_for_improvement
            .map((a) => `<li>${a}</li>`)
            .join("")}</ul>`;
        assessmentContainer.appendChild(areasDiv);

        // Display extracted entities
        const entitiesDiv = document.createElement("div");
        entitiesDiv.className = "mb-4";
        entitiesDiv.innerHTML = "<h5>Extracted Information:</h5>";

        const entityCategories = [
            { key: "age", title: "Age" },
            { key: "gender", title: "Gender" },
            //{ key: "address", title: "Address" },
            { key: "soft_skills", title: "Soft Skills" },
            { key: "hard_skills", title: "Hard Skills" },
            { key: "education_level", title: "Education Level" },
            { key: "course", title: "Course/Major" },
            { key: "experience", title: "Experience" },
            { key: "certification", title: "Certification" },
        ];

        entityCategories.forEach((category) => {
            const values = entities[category.key];

            if (values && values.length > 0) {
                const entityItem = document.createElement("div");
                entityItem.className = "entity-item";

                const entityTitle = document.createElement("div");
                entityTitle.className = "entity-title";
                entityTitle.textContent = category.title;

                const entityValue = document.createElement("div");
                entityValue.className = "entity-value";
                entityValue.textContent = values.join(", ");

                entityItem.appendChild(entityTitle);
                entityItem.appendChild(entityValue);
                entitiesDiv.appendChild(entityItem);
            }
        });

        assessmentContainer.appendChild(entitiesDiv);

        // Recommendation
        // const recommendationDiv = document.createElement("div");
        // recommendationDiv.className = "assessment-recommendation";
        // recommendationDiv.innerHTML = `<strong>Recommendation:</strong> ${assessment.recommendation}`;
        // assessmentContainer.appendChild(recommendationDiv);

        
    }
});
