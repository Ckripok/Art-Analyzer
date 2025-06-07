document.addEventListener("DOMContentLoaded", () => {
  const particlesContainer = document.getElementById("particles");
  const numCircles = 20;
  for (let i = 0; i < numCircles; i++) {
    const circle = document.createElement("div");
    circle.className = "circle";
    const left = Math.floor(Math.random() * 100);
    const delay = (Math.random() * 10).toFixed(1);
    circle.style.left = `${left}%`;
    circle.style.animationDelay = `${delay}s`;
    particlesContainer.appendChild(circle);
  }
});
