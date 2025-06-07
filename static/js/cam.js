document.addEventListener("DOMContentLoaded", () => {
 const selectedImageEl = document.getElementById("selected-image");
 const camImageEl = document.getElementById("cam-image");
 const toggleCamBtn = document.getElementById("toggle-cam");
 let camVisible = false;
 toggleCamBtn.addEventListener("click", () => {
  camVisible = !camVisible;
  camImageEl.style.display = camVisible ? "block" : "none";
  selectedImageEl.style.opacity = camVisible ? "0" : "1";
  toggleCamBtn.title = camVisible ? "Скрыть тепловую карту" : "Показать тепловую карту";
 });
 window.resetCamState = function () {
  camVisible = false;
  camImageEl.style.display = "none";
  selectedImageEl.style.opacity = "0";
  toggleCamBtn.title = "Показать тепловую карту";
 };
});