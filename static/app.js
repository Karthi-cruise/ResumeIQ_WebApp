const form = document.getElementById("analysis-form");
const results = document.getElementById("results");
const ranking = document.getElementById("ranking");
const statusBox = document.getElementById("status");
const resultsEmpty = document.getElementById("results-empty");
const resetButton = document.getElementById("reset-button");
const accountMenuButton = document.getElementById("account-menu-button");
const accountMenu = document.getElementById("account-menu");
const menuLogin = document.getElementById("menu-login");
const menuProfile = document.getElementById("menu-profile");
const menuSaved = document.getElementById("menu-saved");
const authModal = document.getElementById("auth-modal");
const authClose = document.getElementById("auth-close");
const authTitle = document.getElementById("auth-title");
const authCopy = document.getElementById("auth-copy");
const authName = document.getElementById("auth-name");
const authEmail = document.getElementById("auth-email");
const authLogin = document.getElementById("auth-login");
const authLogout = document.getElementById("auth-logout");
const authOpenDestination = document.getElementById("auth-open-destination");
const authLoggedOut = document.getElementById("auth-view-logged-out");
const authLoggedIn = document.getElementById("auth-view-logged-in");
const authUserName = document.getElementById("auth-user-name");
const authUserEmail = document.getElementById("auth-user-email");
const jobDescriptionFileInput = document.getElementById("job-description-file");
const resumeFilesInput = document.getElementById("resume-files");
const jobDescriptionFileLabel = document.getElementById("job-description-file-label");
const resumeFilesLabel = document.getElementById("resume-files-label");
const resumeFilesList = document.getElementById("resume-files-list");
const spotlightEmpty = document.getElementById("spotlight-empty");
const spotlightResults = document.getElementById("spotlight-results");
const spotlightStage = document.getElementById("spotlight-stage");
const spotlightDots = document.getElementById("spotlight-dots");
const spotlightNav = document.getElementById("spotlight-nav");
const spotlightPrev = document.getElementById("spotlight-prev");
const spotlightNext = document.getElementById("spotlight-next");

let selectedResumeFiles = [];
let spotlightItems = [];
let spotlightIndex = 0;
let pendingDestination = "profile";
let authState = loadAuthState();

resetButton.addEventListener("click", () => {
  form.reset();
  selectedResumeFiles = [];
  results.innerHTML = "";
  ranking.classList.add("hidden");
  ranking.innerHTML = "";
  statusBox.textContent = "Add a job description and at least one resume to begin.";
  resultsEmpty.classList.remove("hidden");
  resetSpotlight();
  updateFileSelectionUi();
});

accountMenuButton.addEventListener("click", () => {
  const isOpen = !accountMenu.classList.contains("hidden");
  accountMenu.classList.toggle("hidden", isOpen);
  accountMenuButton.setAttribute("aria-expanded", String(!isOpen));
});

document.addEventListener("click", (event) => {
  if (!accountMenu.contains(event.target) && !accountMenuButton.contains(event.target)) {
    accountMenu.classList.add("hidden");
    accountMenuButton.setAttribute("aria-expanded", "false");
  }
  if (event.target === authModal) {
    closeAuthModal();
  }
});

menuLogin.addEventListener("click", () => openAuthModal("login"));
menuProfile.addEventListener("click", () => openProtectedDestination("profile"));
menuSaved.addEventListener("click", () => openProtectedDestination("saved"));
authClose.addEventListener("click", closeAuthModal);
authLogin.addEventListener("click", handleLogin);
authLogout.addEventListener("click", handleLogout);
authOpenDestination.addEventListener("click", () => {
  closeAuthModal();
  announceDestination(pendingDestination);
});

spotlightPrev.addEventListener("click", () => {
  if (!spotlightItems.length) return;
  spotlightIndex = (spotlightIndex - 1 + spotlightItems.length) % spotlightItems.length;
  renderSpotlightSlide();
});

spotlightNext.addEventListener("click", () => {
  if (!spotlightItems.length) return;
  spotlightIndex = (spotlightIndex + 1) % spotlightItems.length;
  renderSpotlightSlide();
});

jobDescriptionFileInput.addEventListener("change", updateFileSelectionUi);
resumeFilesInput.addEventListener("change", () => {
  const incomingFiles = Array.from(resumeFilesInput.files || []);
  const seen = new Set(selectedResumeFiles.map(fileKey));

  for (const file of incomingFiles) {
    const key = fileKey(file);
    if (!seen.has(key)) {
      selectedResumeFiles.push(file);
      seen.add(key);
    }
  }

  resumeFilesInput.value = "";
  updateFileSelectionUi();
});

updateFileSelectionUi();
resetSpotlight();
refreshAuthUi();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusBox.textContent = "Analyzing resumes against the job description...";
  results.innerHTML = "";
  ranking.classList.add("hidden");
  ranking.innerHTML = "";
  resultsEmpty.classList.add("hidden");

  const formData = new FormData();
  const jdFiles = jobDescriptionFileInput.files;
  const jdText = document.getElementById("job-description").value.trim();
  const resumeText = document.getElementById("resume-text").value.trim();
  const viewMode = document.getElementById("view-mode").value;

  formData.append("job_description", jdText);
  formData.append("resume_text", resumeText);
  formData.append("view_mode", viewMode);

  if (jdFiles.length) {
    formData.append("job_description_file", jdFiles[0]);
  }

  for (const file of selectedResumeFiles) {
    formData.append("resume_files", file);
  }

  if (!jdText && !jdFiles.length) {
    statusBox.textContent = "Please paste a job description or upload a JD file to continue.";
    return;
  }

  if (!selectedResumeFiles.length && !resumeText) {
    statusBox.textContent = "Please upload at least one resume or paste resume text to continue.";
    return;
  }

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Analysis failed.");
    }

    if (authState.loggedIn) {
      storeSavedAnalysis(payload);
    }

    const modeLabel = payload.view_mode === "hiring_manager" ? "Hiring Manager View" : "Candidate View";
    statusBox.textContent = `Analyzed ${payload.analyzed_count} resume${payload.analyzed_count === 1 ? "" : "s"} in ${modeLabel} using ${payload.embedding_provider}, ${payload.vector_backend}, and ${payload.parser_backend}.`;
    resultsEmpty.classList.toggle("hidden", payload.results.length > 0);
    renderRanking(payload.results, payload.view_mode);
    renderSpotlight(payload.results, payload.view_mode);
    payload.results.forEach((item, index) => results.appendChild(renderCard(item, payload.view_mode, index)));
  } catch (error) {
    statusBox.textContent = error.message;
  }
});

function loadAuthState() {
  try {
    return JSON.parse(localStorage.getItem("resumeiq_auth") || "null") || { loggedIn: false, name: "", email: "" };
  } catch {
    return { loggedIn: false, name: "", email: "" };
  }
}

function saveAuthState() {
  localStorage.setItem("resumeiq_auth", JSON.stringify(authState));
}

function refreshAuthUi() {
  if (authState.loggedIn) {
    menuLogin.textContent = "Account";
    authUserName.textContent = authState.name;
    authUserEmail.textContent = authState.email;
    authLoggedOut.classList.add("hidden");
    authLoggedIn.classList.remove("hidden");
    authOpenDestination.textContent = pendingDestination === "saved" ? "Open Saved Analyses" : "Open Profile";
  } else {
    menuLogin.textContent = "Login";
    authLoggedOut.classList.remove("hidden");
    authLoggedIn.classList.add("hidden");
  }
}

function openAuthModal(mode) {
  pendingDestination = mode === "saved" ? "saved" : "profile";
  if (mode === "login") {
    authTitle.textContent = authState.loggedIn ? "Account" : "Login to continue";
    authCopy.textContent = authState.loggedIn
      ? "You are signed in. Open your profile or saved analyses from here."
      : "Login is required to open profile and saved analyses. Resume analysis itself stays available without an account.";
  } else {
    authTitle.textContent = mode === "saved" ? "Saved analyses require login" : "Profile requires login";
    authCopy.textContent = authState.loggedIn
      ? "You are already signed in. Continue to the requested section."
      : "Resume analysis is open to everyone, but account features require sign-in.";
  }
  authName.value = authState.name || "";
  authEmail.value = authState.email || "";
  refreshAuthUi();
  authModal.classList.remove("hidden");
  authModal.setAttribute("aria-hidden", "false");
}

function closeAuthModal() {
  authModal.classList.add("hidden");
  authModal.setAttribute("aria-hidden", "true");
}

function openProtectedDestination(destination) {
  pendingDestination = destination;
  if (!authState.loggedIn) {
    openAuthModal(destination);
    return;
  }
  announceDestination(destination);
}

function handleLogin() {
  const name = authName.value.trim();
  const email = authEmail.value.trim();
  if (!name || !email) {
    authCopy.textContent = "Please enter both name and email to continue.";
    return;
  }
  authState = { loggedIn: true, name, email };
  saveAuthState();
  refreshAuthUi();
  authCopy.textContent = "You are logged in. Account features are now available.";
}

function handleLogout() {
  authState = { loggedIn: false, name: "", email: "" };
  saveAuthState();
  refreshAuthUi();
  authCopy.textContent = "You have been logged out. Resume analysis still works without an account.";
}

function announceDestination(destination) {
  if (destination === "saved") {
    const saved = loadSavedAnalyses();
    statusBox.textContent = saved.length
      ? `Saved analyses available for ${authState.name}: ${saved.length} session${saved.length === 1 ? "" : "s"}.`
      : `No saved analyses yet for ${authState.name}. Run an analysis while logged in to save it.`;
  } else {
    statusBox.textContent = `Profile ready for ${authState.name} (${authState.email}).`;
  }
}

function loadSavedAnalyses() {
  try {
    return JSON.parse(localStorage.getItem("resumeiq_saved_analyses") || "[]");
  } catch {
    return [];
  }
}

function storeSavedAnalysis(payload) {
  const saved = loadSavedAnalyses();
  saved.unshift({
    savedAt: new Date().toISOString(),
    top_resume: payload.top_resume,
    analyzed_count: payload.analyzed_count,
    view_mode: payload.view_mode,
  });
  localStorage.setItem("resumeiq_saved_analyses", JSON.stringify(saved.slice(0, 10)));
}

function fileKey(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function updateFileSelectionUi() {
  const jdFiles = Array.from(jobDescriptionFileInput.files || []);

  jobDescriptionFileLabel.textContent = jdFiles.length ? jdFiles[0].name : "No file chosen";

  if (!selectedResumeFiles.length) {
    resumeFilesLabel.textContent = "No files chosen";
    resumeFilesList.classList.add("hidden");
    resumeFilesList.innerHTML = "";
    return;
  }

  resumeFilesLabel.textContent =
    selectedResumeFiles.length === 1
      ? selectedResumeFiles[0].name
      : `${selectedResumeFiles.length} files selected`;

  resumeFilesList.classList.remove("hidden");
  resumeFilesList.innerHTML = selectedResumeFiles
    .map(
      (file, index) => `
        <button type="button" class="upload-file-chip" data-file-index="${index}" aria-label="Remove ${escapeHtml(file.name)}">
          ${escapeHtml(file.name)}
          <span class="upload-file-remove">×</span>
        </button>
      `
    )
    .join("");

  resumeFilesList.querySelectorAll("[data-file-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.getAttribute("data-file-index"));
      selectedResumeFiles.splice(index, 1);
      updateFileSelectionUi();
    });
  });
}

function resetSpotlight() {
  spotlightItems = [];
  spotlightIndex = 0;
  spotlightEmpty.classList.remove("hidden");
  spotlightResults.classList.add("hidden");
  spotlightStage.innerHTML = "";
  spotlightDots.innerHTML = "";
  spotlightNav.classList.add("hidden");
}

function renderSpotlight(items, viewMode) {
  if (!items || !items.length) {
    resetSpotlight();
    return;
  }

  spotlightItems = items;
  spotlightIndex = 0;
  spotlightEmpty.classList.add("hidden");
  spotlightResults.classList.remove("hidden");
  spotlightNav.classList.toggle("hidden", items.length < 2);
  renderSpotlightSlide(viewMode);
}

function renderSpotlightSlide(viewMode = document.getElementById("view-mode").value) {
  const item = spotlightItems[spotlightIndex];
  if (!item) return;

  const rankLabel = `#${spotlightIndex + 1}`;
  const modeText = viewMode === "hiring_manager" ? "Hiring manager shortlist" : "Candidate comparison";
  const strengths = item.strengths.slice(0, 3);

  spotlightStage.innerHTML = `
    <article class="spotlight-slide ${getScoreClass(item.match_score)}">
      <div class="spotlight-slide-head">
        <div>
          <p class="spotlight-rank">${rankLabel} ranked resume</p>
          <h3>${escapeHtml(item.resume_name)}</h3>
          <p class="spotlight-mode">${modeText}</p>
        </div>
        <div class="spotlight-score ${getScoreClass(item.match_score)}">${item.match_score}%</div>
      </div>
      <p class="spotlight-summary">${escapeHtml(item.summary || item.audience_summary)}</p>
      <div class="spotlight-meta">
        <span>Semantic ${(item.semantic_similarity * 100).toFixed(1)}%</span>
        <span>Coverage ${(item.keyword_coverage * 100).toFixed(1)}%</span>
      </div>
      <div class="spotlight-strengths">
        ${strengths.map((strength) => `<span class="spotlight-chip">${escapeHtml(strength)}</span>`).join("")}
      </div>
    </article>
  `;

  spotlightDots.innerHTML = spotlightItems
    .map(
      (_, index) =>
        `<button type="button" class="spotlight-dot ${index === spotlightIndex ? "active" : ""}" data-spotlight-index="${index}" aria-label="Go to ranked resume ${index + 1}"></button>`
    )
    .join("");

  spotlightDots.querySelectorAll("[data-spotlight-index]").forEach((dot) => {
    dot.addEventListener("click", () => {
      spotlightIndex = Number(dot.getAttribute("data-spotlight-index"));
      renderSpotlightSlide(viewMode);
    });
  });
}

function renderRanking(items, viewMode) {
  if (!items.length) {
    return;
  }
  const heading = items.length > 1 ? "Ranked Resume Results" : "Resume Result";
  const intro =
    viewMode === "hiring_manager"
      ? `Best shortlist candidate: <strong>${escapeHtml(items[0].resume_name)}</strong>`
      : `Best match for this application: <strong>${escapeHtml(items[0].resume_name)}</strong>`;
  ranking.classList.remove("hidden");
  ranking.innerHTML = `
    <h3>${heading}</h3>
    <p>${intro}</p>
    <ol>${items
      .map(
        (item) =>
          `<li>${escapeHtml(item.resume_name)} <strong class="score-text ${getScoreClass(item.match_score)}">${item.match_score}%</strong></li>`
      )
      .join("")}</ol>
  `;
}

function renderCard(item, viewMode, index) {
  const card = document.createElement("article");
  const scoreClass = getScoreClass(item.match_score);
  const modeClass = viewMode === "hiring_manager" ? "mode-hiring" : "mode-candidate";
  const scoreLabel = viewMode === "hiring_manager" ? "Shortlist fit" : "Match readiness";
  card.className = `result-card ${scoreClass} ${modeClass}`;
  card.innerHTML = `
    <div class="result-head">
      <div>
        <div class="mode-row">
          <span class="mode-badge ${modeClass}">${viewMode === "hiring_manager" ? "Hiring Manager View" : "Candidate View"}</span>
          ${viewMode === "hiring_manager" ? `<span class="rank-badge">Rank #${index + 1}</span>` : `<span class="rank-badge candidate">Improvement focus</span>`}
        </div>
        <h3>${escapeHtml(item.resume_name)}</h3>
        <div class="metrics">
          <span>Semantic similarity: ${(item.semantic_similarity * 100).toFixed(1)}%</span>
          <span>Keyword coverage: ${(item.keyword_coverage * 100).toFixed(1)}%</span>
        </div>
      </div>
      <div class="score-pill ${scoreClass}">${item.match_score}% ${scoreLabel}</div>
    </div>

    <p class="summary">${escapeHtml(item.summary)}</p>
    <p class="audience-summary ${modeClass}">${escapeHtml(item.audience_summary)}</p>

    ${viewMode === "hiring_manager" ? renderHiringManagerOverview(item, index) : renderCandidateOverview(item)}
    ${viewMode === "hiring_manager" ? renderHiringFocus(item, index) : renderCandidateFocus(item)}

    <div class="chip-row">
      ${item.matched_skills.map((skill) => `<span class="chip">${escapeHtml(skill)}</span>`).join("")}
      ${item.missing_skills.map((skill) => `<span class="chip missing">${escapeHtml(skill)}</span>`).join("")}
    </div>

    <div class="card-columns ${modeClass}">
      ${viewMode === "hiring_manager"
        ? renderListPanel("Decision Notes", item.hiring_manager_notes.length ? item.hiring_manager_notes : item.strengths)
        : renderListPanel("Strengths", item.strengths)}
      ${renderListPanel(viewMode === "hiring_manager" ? "Risks" : "Gaps", item.gaps)}
      ${renderListPanel(viewMode === "hiring_manager" ? "Recommendation" : "Suggestions", item.suggestions)}
    </div>

    ${viewMode === "candidate" ? renderRewriteSuggestions(item.rewrite_suggestions) : ""}
    ${viewMode === "hiring_manager" ? renderHiringNotes(item.hiring_manager_notes) : ""}

    <div class="requirements">
      <h3>${viewMode === "hiring_manager" ? "Screening Evidence" : "Requirement Evidence"}</h3>
      ${item.matched_requirements
        .map(
          (match) => `
            <div class="requirement ${match.matched ? "matched" : "weak"}">
              <strong>${escapeHtml(match.requirement)}</strong>
              <span>${match.matched ? "Matched" : "Weak match"} • ${(match.score * 100).toFixed(1)}%</span>
              <p>${escapeHtml(match.evidence || "No strong supporting evidence found in the resume.")}</p>
            </div>
          `
        )
        .join("")}
    </div>
  `;
  return card;
}

function renderCandidateOverview(item) {
  const actions = [
    ...(item.gaps || []).slice(0, 2),
    ...(item.suggestions || []).slice(0, 1),
  ].slice(0, 3);
  return `
    <section class="mode-panel candidate-panel">
      <div>
        <p class="mode-panel-label">Candidate action plan</p>
        <h4>What to improve before applying</h4>
      </div>
      <ul class="mode-list">
        ${actions.map((action) => `<li>${escapeHtml(action)}</li>`).join("") || "<li>No critical action items detected.</li>"}
      </ul>
    </section>
  `;
}

function renderCandidateFocus(item) {
  const gapItems = (item.gaps || []).slice(0, 3);
  const actionItems = (item.suggestions || []).slice(0, 3);
  const rewriteItems = (item.rewrite_suggestions || []).slice(0, 2);
  return `
    <section class="focus-grid candidate-focus-grid">
      ${renderFocusTile(
        "Gap watch",
        "Close the most visible misses first.",
        gapItems.length ? gapItems : ["No major JD gaps detected in the current draft."],
        "candidate"
      )}
      ${renderFocusTile(
        "Rewrite next",
        "Make your strongest bullets sound role-specific.",
        rewriteItems.length
          ? rewriteItems.map((entry) => `${entry.requirement}: ${entry.rewritten_bullet}`)
          : ["No rewrite prompts generated for this resume yet."],
        "candidate"
      )}
      ${renderFocusTile(
        "Improvement actions",
        "Use these edits before sending the application.",
        actionItems.length ? actionItems : ["No urgent improvement actions were identified."],
        "candidate"
      )}
    </section>
  `;
}

function renderHiringManagerOverview(item, index) {
  const riskCount = (item.gaps || []).length;
  return `
    <section class="mode-panel hiring-panel">
      <div>
        <p class="mode-panel-label">Shortlist summary</p>
        <h4>${index === 0 ? "Current top-ranked profile" : "Screening comparison candidate"}</h4>
      </div>
      <div class="decision-strip">
        <span class="decision-chip">Rank #${index + 1}</span>
        <span class="decision-chip">${riskCount} risk${riskCount === 1 ? "" : "s"}</span>
        <span class="decision-chip">${item.matched_requirements.filter((match) => match.matched).length} matched requirements</span>
      </div>
    </section>
  `;
}

function renderHiringFocus(item, index) {
  const shortlistNotes = (item.hiring_manager_notes || []).slice(0, 3);
  const risks = (item.gaps || []).slice(0, 3);
  const recommendation = (item.suggestions || []).slice(0, 2);
  const comparisonLead =
    index === 0
      ? "Currently strongest profile in the uploaded shortlist."
      : "Compare this profile against the current top-ranked resume.";
  return `
    <section class="focus-grid hiring-focus-grid">
      ${renderFocusTile(
        "Ranking signal",
        comparisonLead,
        [
          `Rank #${index + 1} by combined semantic and requirement match.`,
          `${item.matched_requirements.filter((match) => match.matched).length} matched requirements highlighted in the evidence section.`,
        ],
        "hiring"
      )}
      ${renderFocusTile(
        "Risk summary",
        "Screening concerns that may affect shortlist confidence.",
        risks.length ? risks : ["No material screening risks detected from the uploaded evidence."],
        "hiring"
      )}
      ${renderFocusTile(
        "Shortlist notes",
        "Notes to carry into comparison or interview review.",
        shortlistNotes.length ? shortlistNotes : recommendation.length ? recommendation : ["No additional shortlist notes were generated."],
        "hiring"
      )}
    </section>
  `;
}

function renderFocusTile(title, eyebrow, items, kind) {
  return `
    <section class="focus-tile ${kind}">
      <p class="focus-eyebrow">${escapeHtml(eyebrow)}</p>
      <h3>${escapeHtml(title)}</h3>
      <ul>
        ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </section>
  `;
}

function renderListPanel(title, items) {
  const content = items.length
    ? `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : "<p>No items.</p>";
  return `<section class="mini-panel"><h3>${title}</h3>${content}</section>`;
}

function renderRewriteSuggestions(items) {
  if (!items || !items.length) {
    return "";
  }
  return `
    <section class="requirements">
      <h3>Tailored Rewrite Suggestions</h3>
      <div class="rewrite-grid">
        ${items
          .map(
            (item) => `
              <div class="rewrite-card">
                <p><strong>Target requirement:</strong> ${escapeHtml(item.requirement)}</p>
                <p><strong>Current bullet:</strong> ${escapeHtml(item.original_bullet)}</p>
                <p><strong>Rewrite:</strong> ${escapeHtml(item.rewritten_bullet)}</p>
                <p><strong>Why:</strong> ${escapeHtml(item.rationale)}</p>
              </div>
            `
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderHiringNotes(items) {
  if (!items || !items.length) {
    return "";
  }
  return `<section class="requirements"><h3>Hiring Manager Notes</h3><ul>${items
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("")}</ul></section>`;
}

function getScoreClass(score) {
  if (score >= 75) return "score-good";
  if (score >= 50) return "score-medium";
  return "score-low";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
