const form = document.getElementById("posting-form");
const result = document.getElementById("result");
const errorBox = document.getElementById("form-error");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;

  // collect text fields
  const data = Object.fromEntries(new FormData(form));

  // checkboxes don't appear in FormData when unchecked, so set them explicitly
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