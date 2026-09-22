const form = document.getElementById("posting-form");
const result = document.getElementById("result");
const errorBox = document.getElementById("form-error");
const imageInput = document.getElementById("image-upload");
const ocrStatus = document.getElementById("ocr-status");

// --- image upload -> OCR -> autofill ---
imageInput.addEventListener("change", async () => {
  const file = imageInput.files[0];
  if (!file) return;

  ocrStatus.hidden = false;
  ocrStatus.textContent = "Reading image...";

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("/extract-text", { method: "POST", body: formData });
    const json = await res.json();

    if (!res.ok) {
      ocrStatus.textContent = json.error;
      return;
    }

    form.elements["title"].value = json.title;
    form.elements["description"].value = json.description;
    ocrStatus.textContent = "Fields filled from image — check them before submitting.";
  } catch (err) {
    ocrStatus.textContent = "Could not read that image.";
  }
});

// --- form submit -> /predict ---
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;

  const data = Object.fromEntries(new FormData(form));

  // unchecked boxes are missing from FormData, so set them explicitly
  ["telecommuting", "has_company_logo", "has_questions"].forEach((name) => {
    data[name] = form.elements[name].checked ? 1 : 0;
  });

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const json = await res.json();

  if (!res.ok) {
    errorBox.textContent = json.error;
    errorBox.hidden = false;
    return;
  }

  document.getElementById("verdict").textContent =
    json.verdict === "fake" ? "Likely fraudulent" : "Looks legitimate";
  document.getElementById("score").textContent = (json.score * 100).toFixed(1) + "%";
  document.getElementById("rf-score").textContent = (json.random_forest_score * 100).toFixed(1) + "%";
  document.getElementById("knn-score").textContent = (json.knn_score * 100).toFixed(1) + "%";

  result.className = json.verdict;
  result.hidden = false;
});