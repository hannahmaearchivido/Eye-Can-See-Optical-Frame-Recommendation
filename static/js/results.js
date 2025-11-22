// ===============================
// POPUP HANDLER (Auto Close + Redirect)
// ===============================
let pendingRedirect = null;
let autoCloseTimer = null;

// ===============================
// BOOTSTRAP 4 MODAL — FRAME SELECT POPULATION
// ===============================
document.addEventListener("DOMContentLoaded", function () {

    $('#frameSelectModal').on('show.bs.modal', function (event) {
  let button = $(event.relatedTarget);

  // Frame data from button
  let frameId = button.data('frame-id');
  let brand = button.data('frame-brand');
  let model = button.data('model-number');
  let color = button.data('frame-color');
  let shape = button.data('frame-shape');
  let price = button.data('price');
  let recommended = button.data('recommended');
  let patientId = button.data('patient-id');
  let imageUrl = button.data('frame-image'); // 🔹 add this attribute to your Select Frame buttons

  // Fill hidden inputs
  $('#modal_frame_id').val(frameId);
  $('#modal_frame_brand').val(brand);
  $('#modal_model_number').val(model);
  $('#modal_frame_color').val(color);
  $('#modal_frame_shape').val(shape);
  $('#modal_price').val(price);
  $('#modal_patient_id').val(patientId);

  // Recommended flag
  $('#is_recommended').val(
    String(recommended).toLowerCase() === "yes" ||
    String(recommended).toLowerCase() === "true"
      ? "Yes"
      : "No"
  );

  // Update modal preview
  $('#previewDetails').text(`${brand} | Model: ${model} | Shape: ${shape} | Color: ${color}`);
  $('#previewPrice').text(`₱${parseFloat(price).toFixed(2)}`);

  // 🔹 Update preview image
  if (imageUrl) {
    $('#previewImage').attr('src', imageUrl).show();
  } else {
    $('#previewImage').hide();
  }
});


});


// ===============================
// FLASH MESSAGES (Bootstrap 4 Compatible)
// ===============================
document.addEventListener("DOMContentLoaded", function () {

    // This block is rendered by Jinja so it must exist before running
    const flashModalEl = document.getElementById("flashModal");
    if (!flashModalEl) return;

    const modal = new bootstrap.Modal(flashModalEl);
    const header = flashModalEl.querySelector(".modal-header");
    const title = flashModalEl.querySelector("#flashModalLabel");
    const body = flashModalEl.querySelector("#flashModalBody");

    // Flash messages populated server-side via Jinja
    const flashMessages = JSON.parse(document.getElementById("flash-data").textContent || "[]");

    flashMessages.forEach(({ category, message }) => {
        const map = {
            success: { icon: "fas fa-check-circle text-success", title: "Success" },
            danger: { icon: "fas fa-exclamation-circle text-danger", title: "Error" },
            warning: { icon: "fas fa-exclamation-triangle text-warning", title: "Warning" },
            info: { icon: "fas fa-info-circle text-primary", title: "Notice" }
        };

        const { icon, title: titleText } = map[category] || map.info;

        header.style.backgroundColor = "#f8f9fc";
        header.style.color = "#212529";

        title.innerHTML = `<i class="${icon} mr-2"></i><span class="font-weight-bold">${titleText}</span>`;
        body.innerHTML = `<p class="mb-5">${message}</p>`;

        // Show modal
        modal.show();

        // Auto close after 3 seconds
        setTimeout(() => modal.hide(), 3000);
    });

});


// ===============================
// BACK TO PAGE 2 BUTTON
// ===============================
document.addEventListener("DOMContentLoaded", function () {
    const backBtn = document.getElementById("btn-back-to-choose-frame");
    if (backBtn) {
        backBtn.addEventListener("click", function () {
            window.location.href = backBtn.dataset.url;
        });
    }
});

