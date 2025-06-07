document.addEventListener("DOMContentLoaded", () => {
 window.addEventListener('DOMContentLoaded', () => {
  document.body.classList.add('loaded');

  const blocks = document.querySelectorAll('.animated-block');
  blocks.forEach((block, i) => {
   setTimeout(() => {
    block.classList.add('visible');
   }, i * 250); // поочередное появление
  });
 });
});