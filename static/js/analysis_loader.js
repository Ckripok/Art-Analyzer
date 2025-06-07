
document.addEventListener("DOMContentLoaded", () => {
 const imageInput = document.getElementById("imageInput");
 const previewScroll = document.getElementById("preview-scroll");
 const resultSection = document.getElementById("result-section");
 const selectedImage = document.getElementById("selected-image");
 const genreResults = document.getElementById("genre-results");
 const styleResults = document.getElementById("style-results");
 const loader = document.getElementById("loader");
 const backgroundBlur = document.getElementById("background-blur");
 let previews = [];
 imageInput.addEventListener("change", () => {
  previewScroll.innerHTML = "";
  previews = [];
  Array.from(imageInput.files).forEach((file, index) => {
   const reader = new FileReader();
   reader.onload = e => {
    const box = document.createElement("div");
    box.className = "image-box";
    const img = document.createElement("img");
    img.src = e.target.result;
    img.className = "preview-image";
    img.onclick = () => {
     previews.forEach(p => p.img.classList.remove("selected"));
     img.classList.add("selected");
     showResult(img.src);
     analyzeImage(file);
     backgroundBlur.style.backgroundImage = `url(${img.src})`;
    };
    const label = document.createElement("small");
    label.className = "filename";
    label.textContent = file.name;
    box.appendChild(img);
    box.appendChild(label);
    previewScroll.appendChild(box);
    previews.push({ img, file });
   };
   reader.readAsDataURL(file);
  });
 });
 function showResult(imgSrc) {
  resultSection.style.display = "block";
  selectedImage.src = imgSrc;
  resetCamState();
 }
 function analyzeImage(file) {
  const genreColors = ['#00ff88', '#00c4ff', '#ffaa00', '#ff5555', '#a0f', '#f0f'];
  const styleColors = ['#00c4ff', '#00ff88', '#ffcc00', '#f06', '#0ff', '#f99'];
  const paletteDiv = document.getElementById("color-palette");
  const formData = new FormData();
  formData.append("file", file);
  paletteDiv.innerHTML = "";
  loader.style.display = "block";
  fetch("/predict_all_combined", {
   method: "POST",
   body: formData
  })
   .then(res => res.json())
   .then(data => {
   genreResults.innerHTML = "";
   styleResults.innerHTML = "";
   // Примеры миниатюр
   window.styleExampleMap = {};
   data.style_top3.forEach(s => {
    window.styleExampleMap[s.label] = s.examples;
   });
   window.genreExampleMap = {};
   data.genre_top3.forEach(g => {
    window.genreExampleMap[g.label] = g.examples;
   });
   // Диаграммы
   renderDonutChart("genreChart", data.genre_top3, genreColors, "Жанры");
   renderDonutChart("styleChart", data.style_top3, styleColors, "Стили");
   // Топ-10 жанров
   data.genre_top3.slice(0, 10).forEach(g => {
    genreResults.innerHTML += `
 <div>
   ${g.label} — ${g.confidence}%
   <div class="bar"><div class="bar-fill" style="width:${g.confidence}%"></div></div>
</div>`;
   });
   // Палитра
   data.palette.forEach(hex => {
    const swatch = document.createElement("div");
    swatch.style.backgroundColor = hex;
    swatch.style.width = "30px";
    swatch.style.height = "30px";
    swatch.style.border = "1px solid #444";
    swatch.style.cursor = "pointer";
    swatch.title = `Нажмите, чтобы скопировать ${hex}`;
    swatch.onclick = () => {
     navigator.clipboard.writeText(hex).then(() => {
      swatch.title = `Скопировано: ${hex}`;
      swatch.style.outline = "2px solid #00ff88";
      setTimeout(() => {
       swatch.title = `Нажмите, чтобы скопировать ${hex}`;
       swatch.style.outline = "none";
      }, 1200);
     });
    };
    paletteDiv.appendChild(swatch);
   });
   // Топ-10 стилей
   data.style_top3.slice(0, 10).forEach(s => {
    styleResults.innerHTML += `
 <div>
   ${s.label} — ${s.confidence}%
   <div class="bar"><div class="bar-fill" style="width:${s.confidence}%"></div></div>
</div>`;
   });
   // CAM-карта
   if (data.cam_path) {
    const camImg = document.getElementById("cam-image");
    camImg.src = `/static/${data.cam_path}?t=${Date.now()}`;
    camImg.style.display = "block";
   }
   // Гистограммы
   if (data.saturation_hist && data.brightness_hist) {
    document.getElementById("saturation-plot").src = `/static/${data.saturation_hist}?t=${Date.now()}`;
    document.getElementById("brightness-plot").src = `/static/${data.brightness_hist}?t=${Date.now()}`;
   }
   if (data.metadata && data.metadata.cdf_path) {
    document.getElementById("cdf-plot").src = `/static/${data.metadata.cdf_path}?t=${Date.now()}`;
   }
   if (data.metadata && data.metadata.pdf_path) {
    document.getElementById("pdf-plot").src = `/static/${data.metadata.pdf_path}?t=${Date.now()}`;
   }
   if (data.metadata && data.metadata.hog_path) {
    document.getElementById("hog-plot").src = `/static/${data.metadata.hog_path}?t=${Date.now()}`;
   }
   if (data.metadata.lbp_path) {
    document.getElementById("lbp-plot").src = `/static/${data.metadata.lbp_path}?t=${Date.now()}`;
   }
   if (data.metadata.orb_path) {
    document.getElementById("orb-plot").src = `/static/${data.metadata.orb_path}?t=${Date.now()}`;
   }
   // Семантический граф
   const genreThreshold = 0.05;
   const styleThreshold = 0.05;
   const allGenres = data.genre_top3_full || data.genre_top3;
   const allStyles = data.style_top3_full || data.style_top3;
   const fullNodes = [
    {
     name: 'Изображение',
     category: 0,
     symbolSize: 100,
     itemStyle: { color: '#ffaa00' }
    }
   ];
   const fullLinks = [];
   allGenres.forEach(g => {
    const conf = parseFloat(g.confidence);
    fullNodes.push({
     name: g.label,
     category: 1,
     symbolSize: conf > genreThreshold ? 45 + conf / 2 : 20,
     value: conf,
     itemStyle: {
      color: conf > genreThreshold ? '#00ff88' : '#555',
      opacity: conf > genreThreshold ? 1 : 0.5
     },
     label: {
      color: conf > genreThreshold ? '#eee' : '#888'
     }
    });
    fullLinks.push({ source: 'Изображение', target: g.label });
   });
   allStyles.forEach(s => {
    const conf = parseFloat(s.confidence);
    fullNodes.push({
     name: s.label,
     category: 2,
     symbolSize: conf > styleThreshold ? 45 + conf / 2 : 20,
     value: conf,
     itemStyle: {
      color: conf > styleThreshold ? '#00c4ff' : '#444',
      opacity: conf > styleThreshold ? 1 : 0.5
     },
     label: {
      color: conf > styleThreshold ? '#eee' : '#888'
     }
    });
    fullLinks.push({ source: 'Изображение', target: s.label });
   });
   const graphChart = echarts.init(document.getElementById('graph-container'));
   // Метаданные
   if (data.metadata) {
    const metaBlock = document.getElementById("image-meta-info");
    metaBlock.innerHTML = "";
    const baseInfo = document.createElement("div");
    baseInfo.style.marginBottom = "12px";
    baseInfo.innerHTML = `
       <p><strong>Размер:</strong> ${data.metadata.width} × ${data.metadata.height}</p>
       <p><strong>Средняя насыщенность:</strong> ${data.metadata.avg_saturation}</p>
       <p><strong>Средняя яркость:</strong> ${data.metadata.avg_brightness}</p>
       <p><strong>Контраст:</strong> ${data.metadata.contrast}</p>`;
    metaBlock.appendChild(baseInfo);
   }
   if (data.metadata && data.metadata.color_statistics) {
    const statBlock = document.createElement("div");
    statBlock.style.marginTop = "20px";
    statBlock.style.background = "#111";
    statBlock.style.border = "1px solid #333";
    statBlock.style.padding = "12px";
    statBlock.style.borderRadius = "6px";
    statBlock.style.color = "#ccc";
    statBlock.style.fontSize = "13px";
    const rgb = data.metadata.color_statistics.rgb;
    const hsv = data.metadata.color_statistics.hsv;
    statBlock.innerHTML = `
       <h3 style="color:#00ff88;">RGB статистика</h3>
       <p><strong>Среднее:</strong> R=${rgb.mean.R.toFixed(4)}, G=${rgb.mean.G.toFixed(4)}, B=${rgb.mean.B.toFixed(4)}</p>
       <p><strong>Дисперсия:</strong> R=${rgb.variance.R.toFixed(4)}, G=${rgb.variance.G.toFixed(4)}, B=${rgb.variance.B.toFixed(4)}</p>
       <p><strong>Корреляции:</strong> R-G=${rgb.correlation.RG.toFixed(4)}, R-B=${rgb.correlation.RB.toFixed(4)}, G-B=${rgb.correlation.GB.toFixed(4)}</p>
       <h3 style="color:#00f395;">HSV статистика</h3>
       <p><strong>Среднее:</strong> H=${hsv.mean.H.toFixed(4)}, S=${hsv.mean.S.toFixed(4)}, V=${hsv.mean.V.toFixed(4)}</p>
       <p><strong>Дисперсия:</strong> H=${hsv.variance.H.toFixed(4)}, S=${hsv.variance.S.toFixed(4)}, V=${hsv.variance.V.toFixed(4)}</p>
       <p><strong>Корреляции:</strong> H-S=${hsv.correlation.HS.toFixed(4)}, H-V=${hsv.correlation.HV.toFixed(4)}, S-V=${hsv.correlation.SV.toFixed(4)}</p>`;
    const metaBlock = document.getElementById("image-meta-info");
    metaBlock.appendChild(statBlock);
   }
   if (data.metadata && data.metadata.brightness_stats) {
    const stats = data.metadata.brightness_stats;
    const statEl = document.getElementById("brightness-statistics");
    statEl.innerHTML = `
       <p><strong>Среднее значение (Mean):</strong> ${stats.mean.toFixed(4)}</p>
       <p><strong>Медиана (Median):</strong> ${stats.median.toFixed(4)}</p>
       <p><strong>Минимум:</strong> ${stats.min.toFixed(4)}</p>
       <p><strong>Максимум:</strong> ${stats.max.toFixed(4)}</p>
       <p><strong>Стандартное отклонение:</strong> ${stats.std.toFixed(4)}</p>
       <p><strong>Мода (наиболее частое значение):</strong> ${stats.mode.toFixed(4)}</p>
       <p><strong>Энергия (сумма квадратов):</strong> ${stats.energy.toFixed(4)}</p>
       <p><strong>Энтропия (сложность распределения):</strong> ${stats.entropy.toFixed(4)}</p>`;
   }
   if (data.metadata && data.metadata.texture_features) {
    const t = data.metadata.texture_features;
    const texBlock = document.createElement("div");
    texBlock.style.marginTop = "20px";
    texBlock.style.background = "#111";
    texBlock.style.border = "1px solid #333";
    texBlock.style.padding = "12px";
    texBlock.style.borderRadius = "6px";
    texBlock.style.color = "#ccc";
    texBlock.style.fontSize = "13px";
    texBlock.innerHTML = `
       <h3 style="color:#ffaa00;">Текстурные характеристики (GLCM)</h3>
       <p><strong>Контраст:</strong> ${t.contrast.toFixed(4)}</p>
       <p><strong>Гомогенность:</strong> ${t.homogeneity.toFixed(4)}</p>
       <p><strong>Энергия:</strong> ${t.energy.toFixed(4)}</p>
       <p><strong>Энтропия:</strong> ${t.entropy.toFixed(4)}</p>
       <p><strong>Корреляция:</strong> ${t.correlation.toFixed(4)}</p>       `;
    document.getElementById("image-meta-info").appendChild(texBlock);
   }
   graphChart.setOption({
    tooltip: { show: false },
    legend: [
     { data: ['Изображение', 'Жанр', 'Стиль'], textStyle: { color: '#aaa' } }
    ],
    series: [{
     type: 'graph',
     layout: 'force',
     roam: true,
     draggable: true,
     label: {
      show: true,
      formatter: '{b}',
      fontSize: 11,
      color: '#eee'
     },
     edgeSymbol: ['circle', 'arrow'],
     edgeSymbolSize: [4, 6],
     force: {
      repulsion: 250,
      edgeLength: [80, 160]
     },
     categories: [
      { name: 'Изображение', itemStyle: { color: '#ffaa00' } },
      { name: 'Жанр', itemStyle: { color: '#00ff88' } },
      { name: 'Стиль', itemStyle: { color: '#00c4ff' } }
     ],
     data: fullNodes,
     links: fullLinks,
     lineStyle: {
      color: 'source',
      curveness: 0.2,
      opacity: 0.6
     }
    }]
   });
   graphChart.on('mouseover', params => {
    if (params.dataType === 'node' && params.data.name !== 'Изображение') {
     const { name, category, value } = params.data;
     const rect = document.getElementById('graph-container').getBoundingClientRect();
     let description = '';
     if (category === 1) {
      description = genreDescriptions[name] || 'Жанр: описание отсутствует.';
     } else if (category === 2) {
      description = styleDescriptions[name] || 'Стиль: описание отсутствует.';
     }
     tooltipEl.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 6px;">${name}</div>
        <div style="margin-bottom: 6px;">Уверенность: ${value.toFixed(2)}%</div>
        <div style="font-size: 12px; color: #aaa;">${description}</div>`;
     tooltipEl.style.left = `${rect.left + params.event.offsetX + 10}px`;
     tooltipEl.style.top = `${rect.top + params.event.offsetY - 20}px`;
     tooltipEl.style.display = 'block';
    }
   });
   graphChart.on('mouseout', () => {
    tooltipEl.style.display = 'none';
   });
   if (data.caption) {
    document.getElementById("caption-result").textContent = data.caption;
   }
  })
   .catch(err => {
   genreResults.innerHTML = "Ошибка анализа.";
   styleResults.innerHTML = "";
   console.error("Ошибка:", err);
  })
   .finally(() => loader.style.display = "none");
 }
 window.addEventListener("load", () => {
  setTimeout(() => {
   const splash = document.getElementById("splash-screen");
   if (splash) splash.style.display = "none";
  }, 3000);
 });
});