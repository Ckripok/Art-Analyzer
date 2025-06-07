document.addEventListener("DOMContentLoaded", () => {
 const modal = document.getElementById("image-modal");
 const modalImg = document.getElementById("modal-img");
 document.body.addEventListener("click", function (e) {
  if (e.target.matches(".examples img")) {
   e.stopPropagation();
   modalImg.src = e.target.src;
   modal.classList.add("show");
  }
 });
 modal.addEventListener("click", function (e) {
  if (e.target === modal) {
   modal.classList.remove("show");
  }
 });
 document.querySelectorAll('.card .examples').forEach(example => {
  example.addEventListener("click", e => {
   e.stopPropagation();
  });
 });
});