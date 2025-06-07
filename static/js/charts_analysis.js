document.addEventListener("DOMContentLoaded", () => {
 let genreChartInstance = null;
 let styleChartInstance = null;
 function renderDonutChart(canvasId, dataList, colors, title) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (canvasId === "genreChart" && genreChartInstance) {
   genreChartInstance.destroy();
  }
  if (canvasId === "styleChart" && styleChartInstance) {
   styleChartInstance.destroy();
  }
  const chart = new Chart(ctx, {
   type: 'doughnut',
   data: {
    labels: dataList.map(item => item.label),
    datasets: [{
     data: dataList.map(item => item.confidence),
     backgroundColor: colors.slice(0, dataList.length),
     borderWidth: 1,
     hoverOffset: 15
    }]
   },
   options: {
    plugins: {
     tooltip: {
      enabled: false,
      external: canvasId === "styleChart" ? customTooltipWithImages : customGenreTooltipWithImages
     },
     legend: {
      display: false
     },
     title: {
      display: false
     }
    },
    layout: {
     padding: 10
    },
    animation: {
     animateScale: true
    }
   }
  });
  if (canvasId === "genreChart") genreChartInstance = chart;
  if (canvasId === "styleChart") styleChartInstance = chart;
 }
 const genreDescriptions = {
  "abstract": "Картины без реалистичных форм, фокус на цвете и форме.",
  "animal-painting": "Изображения животных как основная тема.",
  "cityscape": "Городские пейзажи и архитектурные сцены.",
  "figurative": "Фигуры людей в реалистичном или стилизованном виде.",
  "flower-painting": "Цветы как основной мотив.",
  "genre-painting": "Сценки повседневной жизни.",
  "landscape": "Природные пейзажи, горы, поля, водоёмы.",
  "marina": "Морские сцены и корабли.",
  "mythological-painting": "Мифологические сюжеты и персонажи.",
  "nude-painting-nu": "Обнажённые фигуры, натура.",
  "portrait": "Фокус на одном или нескольких людях.",
  "religious-painting": "Религиозные сюжеты, библейские сцены.",
  "still-life": "Натюрморты: предметы, еда, цветы и пр.",
  "symbolic-painting": "Изображения с глубоким символизмом."
 };
 const styleDescriptions = {
  "Abstract": "Абстрактное искусство без реалистичных форм.",
  "Abstract expressionism": "Эмоциональные мазки, экспрессивный стиль.",
  "Academic": "Классическая академическая живопись.",
  "Art_Nouveau": "Орнаментальный стиль с мотивами природы.",
  "Art_Nouveau_Modern": "Современные версии ар-нуво.",
  "Baroque": "Драматичный, динамичный стиль XVII века.",
  "Constructivism": "Геометрия, индустриальные формы.",
  "Cubism": "Разложение форм на геометрию.",
  "Expressionism": "Передача эмоций через искажение.",
  "Fauvism": "Яркие, упрощённые цветовые пятна.",
  "iconography": "Религиозная и символическая живопись.",
  "Japanese": "Традиционное японское искусство.",
  "Minimalism": "Максимально упрощённые формы.",
  "Naive_Art_Primitivism": "Простой, детский или наивный стиль.",
  "Neoclassicism": "Обращение к античным формам и гармонии.",
  "New_Realism": "Современный реализм.",
  "Northern_Renaissance": "Детализированная живопись Севера Европы.",
  "Pointillism": "Изображение, построенное из цветных точек.",
  "Pop": "Яркие образы массовой культуры.",
  "Primitivism": "Примитивные формы и мотивы.",
  "Realism": "Правдивое отображение реальности.",
  "Renaissance": "Гармония и перспектива эпохи Возрождения.",
  "Rococo": "Лёгкий, изящный, пастельный стиль XVIII века.",
  "Romanticism": "Эмоции, природа, драма.",
  "Symbolism": "Образы с метафорами и тайными смыслами.",
  "Synthetic_Cubism": "Кубизм с коллажами и упрощёнными формами.",
  "Western_Medieval": "Символическая живопись Средневековья."
 };
 function customTooltipWithImages(context) {
  const tooltipModel = context.tooltip;
  const tooltipEl = document.getElementById("graph-info-tooltip");
  if (tooltipModel.opacity === 0) {
   tooltipEl.style.display = 'none';
   return;
  }
  const styleLabel = tooltipModel.dataPoints[0].label;
  const examples = window.styleExampleMap?.[styleLabel] || [];
  let html = `<div style="font-weight: bold; margin-bottom: 6px;">${styleLabel}</div>`;
  html += `<div style="margin-bottom: 6px;">${tooltipModel.dataPoints[0].formattedValue}</div>`;
  const desc = styleDescriptions[styleLabel] || "Описание отсутствует.";
  html += `<div style="font-size: 12px; color: #aaa; margin-bottom: 6px;">${desc}</div>`;
  html += `<div style="display: flex; gap: 6px;">`;
  for (const img of examples) {
   html += `<img src="${img}" style="width: 40px; height: 40px; object-fit: cover; border-radius: 4px; border: 1px solid #333;">`;
  }
  html += `</div>`;
  tooltipEl.innerHTML = html;
  const position = context.chart.canvas.getBoundingClientRect();
  tooltipEl.style.left = position.left + window.pageXOffset + tooltipModel.caretX + 'px';
  tooltipEl.style.top = position.top + window.pageYOffset + tooltipModel.caretY + 'px';
  tooltipEl.style.display = 'block';
 }
 function customGenreTooltipWithImages(context) {
  const tooltipModel = context.tooltip;
  const tooltipEl = document.getElementById('chart-tooltip');
  if (tooltipModel.opacity === 0) {
   tooltipEl.style.display = 'none';
   return;
  }
  const label = tooltipModel.dataPoints[0].label;
  const examples = window.genreExampleMap?.[label] || [];
  let html = `<div style="font-weight: bold; margin-bottom: 6px;">${label}</div>`;
  html += `<div style="margin-bottom: 6px;">${tooltipModel.dataPoints[0].formattedValue}</div>`;
  const desc = genreDescriptions[label] || "Описание отсутствует.";
  html += `<div style="font-size: 12px; color: #aaa; margin-bottom: 6px;">${desc}</div>`;
  html += `<div style="display: flex; gap: 6px;">`;
  for (const img of examples) {
   html += `<img src="${img}" style="width: 40px; height: 40px; object-fit: cover; border-radius: 4px; border: 1px solid #333;">`;
  }
  html += `</div>`;
  tooltipEl.innerHTML = html;
  const position = context.chart.canvas.getBoundingClientRect();
  tooltipEl.style.left = position.left + window.pageXOffset + tooltipModel.caretX + 'px';
  tooltipEl.style.top = position.top + window.pageYOffset + tooltipModel.caretY + 'px';
  tooltipEl.style.display = 'block';
 }
 window.renderDonutChart = renderDonutChart;
});
