function startTransition() {
 document.body.classList.add('fade-out');
 setTimeout(() => {
  window.location.href = "analysis.html";
 }, 600); // Время анимации должно совпадать с CSS
}