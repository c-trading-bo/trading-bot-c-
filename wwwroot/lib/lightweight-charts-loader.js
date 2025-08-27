(function(){
  if (window.LightweightCharts) return;
  var s = document.createElement('script');
  s.src = 'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js';
  s.async = false; // ensure it loads before following inline scripts execute
  document.head.appendChild(s);
})();