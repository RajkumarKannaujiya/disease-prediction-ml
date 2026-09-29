document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });

  document.querySelectorAll(".prediction-form").forEach((form) => {
    const fields = [...form.querySelectorAll("[data-form-field]")];
    const label = form.querySelector("[data-progress-label]");
    const bar = form.querySelector("[role='progressbar']");
    const fill = form.querySelector("[data-progress-fill]");
    const submit = form.querySelector("button[type='submit']");

    const updateProgress = () => {
      let answered = 0;
      fields.forEach((field) => {
        const input = field.querySelector("[data-feature-input]");
        const unknown = field.querySelector("[data-unknown-toggle]");
        const isUnknown = Boolean(unknown?.checked);
        if (input) {
          input.disabled = isUnknown;
          field.classList.toggle("is-unknown", isUnknown);
          if (isUnknown || input.value.trim() !== "") answered += 1;
        }
      });
      if (label) label.textContent = `${answered} of ${fields.length} answered`;
      if (bar) bar.setAttribute("aria-valuenow", String(answered));
      if (fill) fill.style.width = `${fields.length ? (answered / fields.length) * 100 : 0}%`;
    };

    form.addEventListener("input", updateProgress);
    form.addEventListener("change", updateProgress);
    form.addEventListener("reset", () => window.setTimeout(updateProgress, 0));
    form.addEventListener("submit", () => {
      if (submit) {
        submit.disabled = true;
        submit.textContent = "Calculating estimate…";
      }
    });
    updateProgress();
  });
});
