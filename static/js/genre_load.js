document.addEventListener("DOMContentLoaded", () => {
 fetch("/api/genres")
  .then(res => res.json())
  .then(genres => {
  const grid = document.getElementById("genre-grid");
  genres.forEach((genre, index) => {
   const card = document.createElement("div");
   card.className = "card animated-block";
   card.innerHTML = `
                    <img src="${genre.examples[0] || ''}" alt="${genre.name}">
                    <div class="title">${genre.name}</div>
                    <div class="content">
                    <p>${genre.description}</p>
                    <div class="examples">
                    ${genre.examples.map(img => `<img src="${img}">`).join("")}
                    </div>
                    </div>`;
   card.addEventListener("click", () => {
    const isOpen = card.classList.contains("open");
    document.querySelectorAll(".card").forEach(c => c.classList.remove("open"));
    if (!isOpen) card.classList.add("open");
   });
   grid.appendChild(card);
   setTimeout(() => card.classList.add("visible"), index * 25);
  });
 });
});